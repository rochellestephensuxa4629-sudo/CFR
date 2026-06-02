import os
import numpy as np
import nibabel as nib
from PIL import Image
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import cv2
from tqdm import tqdm
import random
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict

class BraTS2021ToDualFormatConverter:
    def __init__(self, data_root, output_dir, selected_modalities=['t1ce', 'flair'], target_size=(256, 256)):
        """
        初始化转换器
        
        Args:
            data_root: BraTS2021数据根目录
            output_dir: 输出目录
            selected_modalities: 选择的模态，必须是两个
            target_size: 目标图像尺寸，默认为256x256
        """
        self.data_root = data_root
        self.output_dir = output_dir
        self.target_size = target_size
        
        if len(selected_modalities) != 2:
            raise ValueError("必须选择两种模态，例如['t1ce', 'flair']")
        
        self.selected_modalities = selected_modalities
        
        # 创建VOC格式目录
        self.voc_dirs = {
            'JPEGImages': os.path.join(output_dir, 'JPEGImages'),
            'Annotations': os.path.join(output_dir, 'Annotations'),
            'ImageSets': os.path.join(output_dir, 'ImageSets', 'Main'),
        }
        
        # 创建训练验证TXT文件目录
        self.train_txt_dir = os.path.join(output_dir, 'train_val_txt')
        
        # 创建所有目录
        for dir_path in self.voc_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        os.makedirs(self.train_txt_dir, exist_ok=True)
        
        # 检测类别定义
        self.classes = {
            0: 'background',
            1: 'necrosis',
            2: 'edema',
            3: 'enhancing_tumor'
        }
        
        # BraTS标签到类别的映射
        self.class_mapping = {
            1: 0,  # necrosis -> class 0
            2: 1,  # edema -> class 1
            4: 2   # enhancing_tumor -> class 2
        }
        
        # 类别名称映射
        self.class_names = ['necrosis', 'edema', 'enhancing_tumor']
        
        self.mode = 'multi_class'  # 默认多类别检测
        
        # 统计信息
        self.stats = {
            'total_cases': 0,
            'total_slices': 0,
            'slices_per_modal': defaultdict(int),
            'bboxes_per_class': defaultdict(int),
            'original_size': None,
            'target_size': target_size
        }
        
        # 用于存储图像信息，便于生成TXT文件
        self.slice_info_by_patient = defaultdict(list)
    
    def load_case_data(self, case_folder):
        """加载单个病例的所有模态数据和分割标签"""
        case_path = os.path.join(self.data_root, case_folder)
        
        if not os.path.exists(case_path):
            return None
        
        files = [f for f in os.listdir(case_path) if f.endswith('.nii.gz')]
        if len(files) < 5:
            return None
        
        data = {}
        
        # 加载所有模态数据
        for modal in self.selected_modalities:
            modal_file = None
            for f in files:
                if modal.lower() in f.lower():
                    modal_file = f
                    break
            
            if not modal_file:
                print(f"Warning: {case_folder} missing {modal}")
                return None
            
            modal_path = os.path.join(case_path, modal_file)
            img = nib.load(modal_path)
            data[modal] = img.get_fdata()
        
        # 加载分割标签
        seg_file = None
        for f in files:
            if 'seg' in f.lower():
                seg_file = f
                break
        
        if not seg_file:
            print(f"Warning: {case_folder} missing segmentation")
            return None
        
        seg_path = os.path.join(case_path, seg_file)
        seg_img = nib.load(seg_path)
        data['segmentation'] = seg_img.get_fdata()
        
        # 记录原始尺寸
        if self.stats['original_size'] is None:
            self.stats['original_size'] = data[self.selected_modalities[0]].shape[:2]
        
        return data
    
    def resample_2d_slice(self, slice_data, is_label=False):
        """
        重采样2D切片到目标尺寸
        
        Args:
            slice_data: 2D切片数据
            is_label: 是否是标签数据（标签使用最近邻插值）
        
        Returns:
            resampled_slice: 重采样后的切片
        """
        if is_label:
            # 标签数据使用最近邻插值，保持类别值不变
            interpolation = cv2.INTER_NEAREST
        else:
            # 图像数据使用线性插值
            interpolation = cv2.INTER_LINEAR
        
        # 重采样到目标尺寸
        resampled = cv2.resize(slice_data, self.target_size, interpolation=interpolation)
        
        return resampled
    
    def extract_2d_bboxes(self, seg_slice):
        """从重采样后的2D分割切片中提取边界框，返回两种格式的数据"""
        bboxes_voc = []  # VOC格式的边界框
        bboxes_txt = []  # TXT格式的边界框字符串
        
        if self.mode == 'single_class':
            # 单类别：检测整体肿瘤
            tumor_mask = (seg_slice > 0).astype(np.uint8)
            contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 10:  # 过滤小区域
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # 确保边界框在图像范围内
                x = max(0, min(x, self.target_size[1] - 1))
                y = max(0, min(y, self.target_size[0] - 1))
                w = min(w, self.target_size[1] - x)
                h = min(h, self.target_size[0] - y)
                
                # VOC格式
                bbox_voc = {
                    'xmin': x,
                    'ymin': y,
                    'xmax': x + w,
                    'ymax': y + h,
                    'class_id': 0,
                    'class_name': 'tumor'
                }
                bboxes_voc.append(bbox_voc)
                
                # TXT格式
                bbox_txt = f"{x},{y},{x + w},{y + h},0"
                bboxes_txt.append(bbox_txt)
                
                # 统计
                self.stats['bboxes_per_class']['tumor'] += 1
        
        else:
            # 多类别：检测不同的肿瘤子区域
            for class_id, class_name in [(1, 'necrosis'), (2, 'edema'), (4, 'enhancing_tumor')]:
                class_mask = (seg_slice == class_id).astype(np.uint8)
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 20:  # 过滤小区域
                        continue
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 确保边界框在图像范围内
                    x = max(0, min(x, self.target_size[1] - 1))
                    y = max(0, min(y, self.target_size[0] - 1))
                    w = min(w, self.target_size[1] - x)
                    h = min(h, self.target_size[0] - y)
                    
                    # 映射到类别
                    if class_id == 1:
                        mapped_class = 0  # necrosis -> class 0
                        mapped_name = 'necrosis'
                    elif class_id == 2:
                        mapped_class = 1  # edema -> class 1
                        mapped_name = 'edema'
                    elif class_id == 4:
                        mapped_class = 2  # enhancing_tumor -> class 2
                        mapped_name = 'enhancing_tumor'
                    else:
                        continue
                    
                    # VOC格式
                    bbox_voc = {
                        'xmin': x,
                        'ymin': y,
                        'xmax': x + w,
                        'ymax': y + h,
                        'class_id': mapped_class,
                        'class_name': mapped_name
                    }
                    bboxes_voc.append(bbox_voc)
                    
                    # TXT格式
                    bbox_txt = f"{x},{y},{x + w},{y + h},{mapped_class}"
                    bboxes_txt.append(bbox_txt)
                    
                    # 统计
                    self.stats['bboxes_per_class'][mapped_name] += 1
        
        return bboxes_voc, bboxes_txt
    
    def normalize_modal_slice(self, slice_data):
        """标准化单个模态的2D切片"""
        # 去除背景（零值区域）
        mask = slice_data > 0
        if np.sum(mask) > 0:
            # 只对非零区域进行标准化
            non_zero = slice_data[mask]
            mean = np.mean(non_zero)
            std = np.std(non_zero)
            if std > 0:
                slice_data = (slice_data - mean) / std
        
        # 缩放到0-255范围
        slice_data = np.clip(slice_data, -3, 3)  # 截断极端值
        slice_data = (slice_data + 3) / 6 * 255  # 映射到0-255
        slice_data = slice_data.astype(np.uint8)
        
        return slice_data
    
    def generate_filename(self, case_folder, slice_idx, modal, extension='.jpg'):
        """生成包含模态信息的文件名"""
        # 格式: BraTS2021_00000_slice_080_t1ce.jpg
        return f"{case_folder}_slice_{slice_idx:03d}_{modal}{extension}"
    
    def create_voc_xml(self, image_filename, image_size, bboxes_voc, output_xml_path, modal):
        """创建标准的VOC格式XML文件"""
        root = ET.Element('annotation')
        
        # 文件夹
        folder = ET.SubElement(root, 'folder')
        folder.text = 'JPEGImages'
        
        # 文件名
        filename = ET.SubElement(root, 'filename')
        filename.text = image_filename
        
        # 来源
        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'BraTS2021 Database'
        
        # 图像尺寸信息（包含重采样信息）
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(image_size[1])
        height = ET.SubElement(size, 'height')
        height.text = str(image_size[0])
        depth = ET.SubElement(size, 'depth')
        depth.text = '1'  # 灰度图
        
        # 重采样信息
        resample_info = ET.SubElement(root, 'resample_info')
        original_size = ET.SubElement(resample_info, 'original_size')
        original_size.text = f"{self.stats['original_size'][1]}x{self.stats['original_size'][0]}"
        target_size_elem = ET.SubElement(resample_info, 'target_size')
        target_size_elem.text = f"{self.target_size[1]}x{self.target_size[0]}"
        
        # 模态信息
        modal_info = ET.SubElement(root, 'modal')
        modal_info.text = modal
        
        # 是否分割
        segmented = ET.SubElement(root, 'segmented')
        segmented.text = '0'
        
        # 每个对象
        for bbox in bboxes_voc:
            obj = ET.SubElement(root, 'object')
            
            # 类别名称
            name = ET.SubElement(obj, 'name')
            name.text = bbox['class_name']
            
            # 姿态
            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            
            # 是否截断
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'
            
            # 是否困难
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '0'
            
            # 边界框
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(bbox['xmin'])
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(bbox['ymin'])
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(bbox['xmax'])
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(bbox['ymax'])
        
        # 美化XML并保存
        xml_str = ET.tostring(root, encoding='utf-8')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent='  ')
        
        with open(output_xml_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
    
    def process_single_slice(self, case_folder, slice_idx, data, split_name):
        """
        处理单个切片，重采样到256x256，生成两种格式的数据
        
        流程：
        1. 获取原始分割切片
        2. 重采样分割切片到256x256
        3. 在重采样后的分割切片上提取边界框
        4. 为每个模态重采样图像并保存
        """
        # 获取原始分割切片
        seg_slice_original = data['segmentation'][:, :, slice_idx]
        
        # 重采样分割切片到256x256
        seg_slice_resampled = self.resample_2d_slice(seg_slice_original, is_label=True)
        
        # 跳过没有肿瘤的切片
        if np.sum(seg_slice_resampled > 0) < 10:
            return None
        
        # 在重采样后的分割切片上提取边界框（两种格式）
        bboxes_voc, bboxes_txt = self.extract_2d_bboxes(seg_slice_resampled)
        if not bboxes_voc:
            return None
        
        # 存储图像路径和边界框信息，用于生成TXT文件
        image_paths = []
        image_info = {
            'case_id': case_folder,
            'slice_idx': slice_idx,
            'split': split_name,
            'bboxes_txt': bboxes_txt,
            'original_size': seg_slice_original.shape,
            'resampled_size': seg_slice_resampled.shape
        }
        
        # 为每个模态生成图像和VOC格式文件
        for modal in self.selected_modalities:
            # 获取当前模态的原始切片数据
            modal_slice_original = data[modal][:, :, slice_idx]
            
            # 重采样模态切片到256x256
            modal_slice_resampled = self.resample_2d_slice(modal_slice_original, is_label=False)
            
            # 标准化重采样后的图像
            modal_slice_normalized = self.normalize_modal_slice(modal_slice_resampled)
            
            # 转换为PIL图像（灰度图）
            image = Image.fromarray(modal_slice_normalized).convert('L')
            
            # 生成文件名
            image_filename = self.generate_filename(case_folder, slice_idx, modal, '.jpg')
            xml_filename = self.generate_filename(case_folder, slice_idx, modal, '.xml')
            
            # 保存图像（256x256）
            image_path = os.path.join(self.voc_dirs['JPEGImages'], image_filename)
            image.save(image_path, quality=95)
            
            # 保存VOC XML标注
            xml_path = os.path.join(self.voc_dirs['Annotations'], xml_filename)
            image_size = image.size[::-1]  # (height, width)，应该是(256, 256)
            self.create_voc_xml(image_filename, image_size, bboxes_voc, xml_path, modal)
            
            # 保存绝对路径用于TXT文件
            abs_image_path = os.path.abspath(image_path)
            image_paths.append(abs_image_path)
            
            # 统计
            self.stats['slices_per_modal'][modal] += 1
        
        # 生成TXT文件的一行
        txt_line = f"{image_paths[0]} {image_paths[1]} {' '.join(bboxes_txt)}"
        
        # 存储信息
        image_info['txt_line'] = txt_line
        image_info['image_id'] = f"{case_folder}_slice_{slice_idx:03d}"
        
        self.stats['total_slices'] += 1
        
        return image_info
    
    def visualize_resample_effect(self, case_folder, slice_idx, data):
        """可视化重采样效果，用于调试"""
        import matplotlib.pyplot as plt
        
        # 获取原始数据
        seg_slice_original = data['segmentation'][:, :, slice_idx]
        modal_slice_original = data[self.selected_modalities[0]][:, :, slice_idx]
        
        # 重采样
        seg_slice_resampled = self.resample_2d_slice(seg_slice_original, is_label=True)
        modal_slice_resampled = self.resample_2d_slice(modal_slice_original, is_label=False)
        modal_slice_normalized = self.normalize_modal_slice(modal_slice_resampled)
        
        # 提取边界框
        bboxes_voc, _ = self.extract_2d_bboxes(seg_slice_resampled)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始分割
        axes[0, 0].imshow(seg_slice_original, cmap='gray')
        axes[0, 0].set_title(f'Original Segmentation\n{seg_slice_original.shape}')
        axes[0, 0].axis('off')
        
        # 重采样分割
        axes[0, 1].imshow(seg_slice_resampled, cmap='gray')
        axes[0, 1].set_title(f'Resampled Segmentation\n{seg_slice_resampled.shape}')
        axes[0, 1].axis('off')
        
        # 边界框可视化
        seg_with_bbox = cv2.cvtColor(seg_slice_resampled.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        for bbox in bboxes_voc:
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            cv2.rectangle(seg_with_bbox, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        axes[0, 2].imshow(seg_with_bbox)
        axes[0, 2].set_title(f'Bounding Boxes\n{len(bboxes_voc)} boxes')
        axes[0, 2].axis('off')
        
        # 原始模态
        axes[1, 0].imshow(modal_slice_original, cmap='gray')
        axes[1, 0].set_title(f'Original {self.selected_modalities[0]}\n{modal_slice_original.shape}')
        axes[1, 0].axis('off')
        
        # 重采样模态
        axes[1, 1].imshow(modal_slice_resampled, cmap='gray')
        axes[1, 1].set_title(f'Resampled {self.selected_modalities[0]}\n{modal_slice_resampled.shape}')
        axes[1, 1].axis('off')
        
        # 标准化模态
        axes[1, 2].imshow(modal_slice_normalized, cmap='gray')
        axes[1, 2].set_title(f'Normalized {self.selected_modalities[0]}\n{modal_slice_normalized.shape}')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Resample Effect: {case_folder} slice {slice_idx}\n{self.stats["original_size"]} -> {self.target_size}', fontsize=16)
        plt.tight_layout()
        
        # 保存图像
        viz_dir = os.path.join(self.output_dir, 'visualization')
        os.makedirs(viz_dir, exist_ok=True)
        viz_path = os.path.join(viz_dir, f'{case_folder}_slice_{slice_idx:03d}_resample.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"重采样可视化已保存: {viz_path}")
        
        return viz_path
    
    def process_cases_by_split(self, case_list, split_name, visualize_first_n=0):
        """处理指定病例列表"""
        split_info = []
        
        for case_idx, case_folder in enumerate(tqdm(case_list, desc=f"处理{split_name}集")):
            data = self.load_case_data(case_folder)
            if data is None:
                continue
            
            seg_3d = data['segmentation']
            depth = seg_3d.shape[2]
            
            # 每个病例选择部分切片，避免数据量过大
            # slice_indices = list(range(depth))
            # if depth > 20:  # 如果切片太多，随机选择一部分
            #     slice_indices = random.sample(slice_indices, min(20, depth))
            slice_indices = []
            for z in range(depth):
                seg_slice_original = data['segmentation'][:, :, z]
                # 只选择包含肿瘤的切片
                if np.sum(seg_slice_original > 0) > 0:  # 只要有肿瘤像素
                    slice_indices.append(z)
            
            print(f"病例 {case_folder}: {len(slice_indices)}/{depth} 个有肿瘤的切片")
            
            case_slices = []
            for slice_idx, z in enumerate(slice_indices):
                try:
                    # 可视化前几个切片的重采样效果
                    if case_idx < visualize_first_n and slice_idx == 0:
                        self.visualize_resample_effect(case_folder, z, data)
                    
                    result = self.process_single_slice(case_folder, z, data, split_name)
                    if result:
                        split_info.append(result)
                        case_slices.append(result['image_id'])
                except Exception as e:
                    print(f"处理切片 {case_folder}_slice_{z} 时出错: {str(e)}")
                    continue
            
            # 存储每个病例的切片信息
            if case_slices:
                self.slice_info_by_patient[case_folder] = case_slices
                self.stats['total_cases'] += 1
        
        return split_info
    
    def save_voc_imagesets(self, all_splits_info):
        """创建VOC格式的ImageSets文件"""
        # 创建全局划分文件（所有模态）
        for split_name, split_info in all_splits_info.items():
            split_file = os.path.join(self.voc_dirs['ImageSets'], f'{split_name}.txt')
            with open(split_file, 'w') as f:
                for info in split_info:
                    f.write(f"{info['image_id']}\n")
        
        # 创建trainval文件（训练+验证）
        train_info = all_splits_info.get('train', [])
        val_info = all_splits_info.get('val', [])
        trainval_file = os.path.join(self.voc_dirs['ImageSets'], 'trainval.txt')
        with open(trainval_file, 'w') as f:
            for info in train_info + val_info:
                f.write(f"{info['image_id']}\n")
    
    def save_train_val_txt(self, all_splits_info):
        """保存训练验证TXT文件（2007_train.txt格式）"""
        for split_name, split_info in all_splits_info.items():
            txt_file = os.path.join(self.train_txt_dir, f'2007_{split_name}.txt')
            
            with open(txt_file, 'w') as f:
                for info in split_info:
                    f.write(f"{info['txt_line']}\n")
            
            print(f"保存 {split_name} 集TXT文件: {txt_file}, {len(split_info)} 行")
    
    def create_dataset_by_patient_split(self, test_ratio=0.15, val_ratio=0.15, visualize_first_n=2):
        """按病人划分数据集并处理"""
        # 获取所有病例文件夹
        case_folders = sorted([d for d in os.listdir(self.data_root) 
                              if os.path.isdir(os.path.join(self.data_root, d)) 
                              and d.startswith('BraTS2021_')])
        
        print(f"找到 {len(case_folders)} 个病例")
        print(f"重采样目标尺寸: {self.target_size}")
        
        # 按病人划分数据集
        train_cases, test_cases = train_test_split(
            case_folders, test_size=test_ratio, random_state=42
        )
        train_cases, val_cases = train_test_split(
            train_cases, test_size=val_ratio/(1-test_ratio), random_state=42
        )
        
        print(f"训练集病例数: {len(train_cases)}")
        print(f"验证集病例数: {len(val_cases)}")
        print(f"测试集病例数: {len(test_cases)}")
        
        # 保存病例划分
        split_info = {
            'train': train_cases,
            'val': val_cases,
            'test': test_cases,
            'total_cases': len(case_folders),
            'split_ratios': {
                'train': len(train_cases) / len(case_folders),
                'val': len(val_cases) / len(case_folders),
                'test': len(test_cases) / len(case_folders)
            },
            'resample_info': {
                'original_size': self.stats['original_size'],
                'target_size': self.target_size
            }
        }
        
        with open(os.path.join(self.output_dir, 'patient_split.json'), 'w') as f:
            json.dump(split_info, f, indent=4)
        
        # 处理每个分区的病例
        all_splits_info = {}
        
        print("\n处理训练集病例...")
        train_info = self.process_cases_by_split(train_cases, 'train', visualize_first_n)
        all_splits_info['train'] = train_info
        
        print("\n处理验证集病例...")
        val_info = self.process_cases_by_split(val_cases, 'val')
        all_splits_info['val'] = val_info
        
        print("\n处理测试集病例...")
        test_info = self.process_cases_by_split(test_cases, 'test')
        all_splits_info['test'] = test_info
        
        # 保存VOC格式的ImageSets文件
        self.save_voc_imagesets(all_splits_info)
        
        # 保存训练验证TXT文件
        self.save_train_val_txt(all_splits_info)
        
        # 保存统计信息
        self.save_statistics()
        
        return all_splits_info
    
    def save_statistics(self):
        """保存统计信息"""
        stats_file = os.path.join(self.output_dir, 'dataset_statistics.json')
        
        # 计算平均每张图像的边界框数
        if self.stats['total_slices'] > 0:
            total_bboxes = sum(self.stats['bboxes_per_class'].values())
            avg_bboxes_per_slice = total_bboxes / self.stats['total_slices']
            self.stats['avg_bboxes_per_slice'] = avg_bboxes_per_slice
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=4)
        
        print("\n" + "="*60)
        print("数据集统计信息:")
        print(f"原始图像尺寸: {self.stats['original_size']}")
        print(f"重采样目标尺寸: {self.target_size}")
        print(f"总病例数: {self.stats['total_cases']}")
        print(f"总切片数: {self.stats['total_slices']}")
        print("各模态切片数:")
        for modal, count in self.stats['slices_per_modal'].items():
            print(f"  {modal}: {count}")
        print("各类别边界框数:")
        for class_name, count in self.stats['bboxes_per_class'].items():
            print(f"  {class_name}: {count}")
        if 'avg_bboxes_per_slice' in self.stats:
            print(f"平均每切片边界框数: {self.stats['avg_bboxes_per_slice']:.2f}")
        print("="*60)
    
    def save_class_names(self):
        """保存类别名称文件"""
        class_file = os.path.join(self.output_dir, 'classes.txt')
        
        with open(class_file, 'w') as f:
            if self.mode == 'single_class':
                f.write("tumor\n")
            else:
                for class_name in self.class_names:
                    f.write(f"{class_name}\n")
    
    def create_sample_demo(self):
        """创建示例文件，展示生成的格式"""
        # 读取训练TXT文件的前几行作为示例
        train_txt_file = os.path.join(self.train_txt_dir, '2007_train.txt')
        
        if os.path.exists(train_txt_file):
            with open(train_txt_file, 'r') as f:
                lines = f.readlines()[:3]
            
            # 保存示例
            sample_file = os.path.join(self.output_dir, 'sample_format.txt')
            with open(sample_file, 'w') as f:
                f.write(f"[file name]: 2007_train.txt\n")
                f.write(f"[file content begin]\n")
                for line in lines:
                    f.write(line)
                f.write(f"[file content end]\n")
            
            # 读取VOC格式示例
            annotations_dir = self.voc_dirs['Annotations']
            xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
            if xml_files:
                sample_xml = os.path.join(annotations_dir, xml_files[0])
                with open(sample_xml, 'r') as f:
                    xml_content = f.read()
                
                xml_sample_file = os.path.join(self.output_dir, 'sample_voc_format.xml')
                with open(xml_sample_file, 'w') as f:
                    f.write(xml_content)
            
            print(f"\n格式示例已保存:")
            print(f"  TXT格式示例: {sample_file}")
            print(f"  VOC XML示例: {xml_sample_file}")
            
            # 打印示例内容
            print("\n" + "="*80)
            print("TXT文件格式示例（2007_train.txt）:")
            for i, line in enumerate(lines, 1):
                print(f"行{i}: {line.strip()}")
                if i == 1:  # 解析第一行
                    parts = line.strip().split()
                    print(f"  模态1路径: {parts[0]}")
                    print(f"  模态2路径: {parts[1]}")
                    print(f"  边界框数量: {len(parts) - 2}")
                    if len(parts) > 2:
                        print(f"  第一个边界框: {parts[2]}")
            print("="*80)


def main():
    # 配置路径
    data_root = "/root/autodl-tmp/detection/model/yolox-pytorch-main/BraTS2021"
    output_dir = "/root/autodl-tmp/detection/model/yolox-pytorch-main/BraTS2021_256x256_2"
    
    # 选择要处理的模态（必须是两种）
    selected_modalities = ['t1ce', 'flair']
    
    # 设置重采样目标尺寸
    target_size = (256, 256)  # (height, width)
    
    print(f"开始转换BraTS2021数据集（重采样到{target_size}）")
    print(f"数据目录: {data_root}")
    print(f"输出目录: {output_dir}")
    print(f"选择的模态: {selected_modalities}")
    print(f"重采样尺寸: {target_size}")
    print(f"检测模式: 多类别检测")
    print(f"将生成两种格式:")
    print(f"  1. 标准VOC格式")
    print(f"  2. 训练验证TXT文件（2007_train.txt格式）")
    print("-" * 80)
    
    # 创建转换器实例
    converter = BraTS2021ToDualFormatConverter(
        data_root=data_root,
        output_dir=output_dir,
        selected_modalities=selected_modalities,
        target_size=target_size
    )
    
    # 设置检测模式
    converter.mode = 'multi_class'  # 可选: 'multi_class' 或 'single_class'
    
    # 按病人划分并处理数据
    all_splits_info = converter.create_dataset_by_patient_split(
        test_ratio=0.15,
        val_ratio=0.15,
        visualize_first_n=2  # 可视化前2个病例的重采样效果
    )
    
    # 保存类别名称
    converter.save_class_names()
    
    # 生成示例文件
    converter.create_sample_demo()
    
    # 打印最终信息
    print(f"\n数据集转换完成！")
    print(f"输出目录结构:")
    print(f"  {output_dir}/")
    print(f"  ├── JPEGImages/              # 图像文件（256x256，.jpg格式）")
    print(f"  ├── Annotations/             # XML标注文件（基于256x256图像）")
    print(f"  ├── ImageSets/Main/          # VOC数据集划分文件")
    print(f"  ├── train_val_txt/           # 训练验证TXT文件")
    print(f"  ├── visualization/           # 重采样效果可视化（可选）")
    print(f"  ├── classes.txt              # 类别名称")
    print(f"  ├── patient_split.json       # 病人划分信息（包含重采样信息）")
    print(f"  ├── dataset_statistics.json  # 数据集统计")
    print(f"  ├── sample_format.txt        # TXT格式示例")
    print(f"  └── sample_voc_format.xml    # VOC XML格式示例")
    
    # 验证文件生成情况
    print(f"\n文件生成验证:")
    
    # 检查VOC格式文件
    jpeg_dir = converter.voc_dirs['JPEGImages']
    annotations_dir = converter.voc_dirs['Annotations']
    
    jpeg_files = [f for f in os.listdir(jpeg_dir) if f.endswith('.jpg')]
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    
    if jpeg_files:
        # 检查图像尺寸
        sample_image_path = os.path.join(jpeg_dir, jpeg_files[0])
        with Image.open(sample_image_path) as img:
            img_size = img.size  # (width, height)
            print(f"  ✓ 图像尺寸验证: {img_size} (应为{target_size[1]}x{target_size[0]})")
    
    # 检查边界框坐标是否在合理范围内
    train_txt_file = os.path.join(converter.train_txt_dir, '2007_train.txt')
    if os.path.exists(train_txt_file):
        with open(train_txt_file, 'r') as f:
            first_line = f.readline().strip()
        
        parts = first_line.split()
        if len(parts) > 2:
            # 检查第一个边界框
            bbox_str = parts[2]
            xmin, ymin, xmax, ymax, class_id = map(int, bbox_str.split(','))
            
            print(f"  ✓ 边界框坐标验证:")
            print(f"    第一个边界框: ({xmin}, {ymin}, {xmax}, {ymax}, class={class_id})")
            print(f"    图像尺寸: {target_size}")
            
            # 检查边界框是否在图像范围内
            if 0 <= xmin < xmax <= target_size[1] and 0 <= ymin < ymax <= target_size[0]:
                print(f"    ✓ 边界框在图像范围内")
            else:
                print(f"    ⚠ 边界框超出图像范围")


if __name__ == "__main__":
    main()