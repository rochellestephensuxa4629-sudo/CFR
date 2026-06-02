import json
import os
import nibabel as nib
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import random
from collections import defaultdict
import argparse

def nii_to_image(nii_path, output_dir, base_name, modality="ct"):
    # CT使用窗宽窗位，PET直接全局归一化
    if modality == "ct":
        ww = 1500
        wl = -600
    elif modality == "pet":
        # PET直接归一化，不再设置窗宽窗位
        pass
    try:
        # 读取nii文件
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # 如果是3D数据，取中间层
        if len(data.shape) == 3:
            slice_idx = data.shape[2] // 2
            slice_data = data[:, :, slice_idx]
        else:
            slice_data = data

        if modality == "ct":
            # CT：窗宽窗位截断 + 线性映射到0-255
            low = wl - ww / 2.0
            high = wl + ww / 2.0
            slice_data = np.clip(slice_data, low, high)
            if high > low:
                slice_data = (slice_data - low) / (high - low) * 255
            slice_data = slice_data.astype(np.uint8)
            # 二次归一化（保持原有行为）
            if slice_data.max() > slice_data.min():
                slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
            slice_data = slice_data.astype(np.uint8)
        else:  # modality == "pet"
            # PET：直接全局归一化到0-255
            if slice_data.max() > slice_data.min():
                slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
            else:
                slice_data = np.zeros_like(slice_data)
            slice_data = slice_data.astype(np.uint8)
        
        # 保存为PNG
        output_path = os.path.join(output_dir, f"{base_name}_{modality}.png")
        cv2.imwrite(output_path, slice_data)
        return output_path
    except Exception as e:
        print(f"Error processing {nii_path}: {e}")
        return None

def mask_to_xml(mask_path, output_dir, base_name, image_width, image_height):
    """从mask生成VOC格式的XML标注文件（CT和PET共用）"""
    try:
        # 读取mask文件
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata()
        
        # 如果是3D数据，取中间层
        if len(mask_data.shape) == 3:
            slice_idx = mask_data.shape[2] // 2
            slice_mask = mask_data[:, :, slice_idx]
        else:
            slice_mask = mask_data
        
        # 二值化处理
        slice_mask = (slice_mask > 0).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建XML根元素
        annotation = ET.Element("annotation")
        
        # 添加文件夹和文件名
        folder = ET.SubElement(annotation, "folder")
        folder.text = "VOC_CT_PET"
        
        filename = ET.SubElement(annotation, "filename")
        filename.text = f"{base_name}.png"  # 统一使用base_name，不区分模态
        
        # 添加图像尺寸
        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(image_width)
        height = ET.SubElement(size, "height")
        height.text = str(image_height)
        depth = ET.SubElement(size, "depth")
        depth.text = "1"
        
        bbox_list = []  # 存储所有bbox
        
        # 添加每个检测目标
        for i, contour in enumerate(contours):
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 忽略太小的区域
            if w < 5 or h < 5:
                continue
                
            # 创建object元素
            obj = ET.SubElement(annotation, "object")
            
            name = ET.SubElement(obj, "name")
            name.text = "lesion"
            
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            
            # 添加边界框
            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(x)
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(y)
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(x + w)
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(y + h)
            
            # 添加到bbox列表
            bbox_list.append((x, y, x + w, y + h))
        
        # 保存XML文件
        xml_str = ET.tostring(annotation, encoding='unicode')
        xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
        
        # 只生成一个XML文件，不区分模态
        xml_path = os.path.join(output_dir, f"{base_name}.xml")
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        
        return xml_path, len(bbox_list), bbox_list
        
    except Exception as e:
        print(f"Error processing mask {mask_path}: {e}")
        return None, 0, []

def extract_patient_id(file_path):
    """从文件路径中提取病人ID，格式为ct_path[23:32]"""
    if not file_path or len(file_path) < 32:
        return None
    return file_path[23:32]

def split_by_patient_fixed(patient_samples, seed=42):
    """按病人划分训练集和验证集，固定63:8比例"""
    random.seed(seed)
    patient_ids = list(patient_samples.keys())
    random.shuffle(patient_ids)
    
    # 固定63:8比例
    train_patients = patient_ids[:63] if len(patient_ids) >= 71 else patient_ids[:int(len(patient_ids) * 0.887)]
    val_patients = patient_ids[63:71] if len(patient_ids) >= 71 else patient_ids[int(len(patient_ids) * 0.887):]
    
    return train_patients, val_patients

def split_by_sample_random(all_samples, train_ratio=0.8, seed=42):
    """随机划分训练集和验证集（不考虑病人ID）"""
    random.seed(seed)
    random.shuffle(all_samples)
    
    train_count = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:train_count]
    val_samples = all_samples[train_count:]
    
    return train_samples, val_samples

def convert_to_voc(json_path, output_base_dir, image_size=(512, 512), 
                   process_pet=True, split_by_patient_flag=True, 
                   train_ratio=0.8, seed=42):
    """
    主转换函数
    
    参数:
    json_path: JSON文件路径
    output_base_dir: 输出目录
    image_size: 图像尺寸
    process_pet: 是否处理PET图像 (True/False)
    split_by_patient_flag: 是否按病人划分 (True:按病人划分, False:随机划分)
    train_ratio: 训练集比例 (0-1之间)，仅在随机划分时使用
    seed: 随机种子
    """
    # 创建输出目录
    images_dir = os.path.join(output_base_dir, "JPEGImages")
    annotations_dir = os.path.join(output_base_dir, "Annotations")
    image_sets_dir = os.path.join(output_base_dir, "ImageSets/Main")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(image_sets_dir, exist_ok=True)
    
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 合并所有样本
    all_samples = data.get("training", []) + data.get("validation", [])
    
    print(f"Total samples in JSON: {len(all_samples)}")
    
    # 根据split_by_patient_flag选择划分策略
    if split_by_patient_flag:
        print("Splitting by patient ID (fixed 63:8 ratio)...")
        # 按病人分组
        patient_samples = defaultdict(list)
        
        for idx, sample in enumerate(all_samples):
            ct_path = sample.get("ct_path", "")
            pet_path = sample.get("pet_path", "")
            mask_path = sample.get("mask_path", "")
            
            # 提取病人ID
            patient_id = extract_patient_id(ct_path)
            if patient_id:
                # 为样本分配唯一ID，基于病人和切片索引
                sample_id = f"patient_{patient_id}_sample_{idx:04d}"
                
                patient_samples[patient_id].append({
                    "sample_id": sample_id,
                    "ct_path": ct_path,
                    "pet_path": pet_path,
                    "mask_path": mask_path,
                    "idx": idx
                })
        
        print(f"Found {len(patient_samples)} unique patients")
        
        # 按病人划分训练验证集，固定63:8比例
        train_patients, val_patients = split_by_patient_fixed(patient_samples, seed)
        
        print(f"Training patients: {len(train_patients)} (63)")
        print(f"Validation patients: {len(val_patients)} (8)")
        
        # 准备训练和验证文件内容
        train_lines = []
        val_lines = []
        train_samples_list = []
        val_samples_list = []
        
        total_objects = 0
        processed_train = 0
        processed_val = 0
        
        # 处理所有病人
        for patient_id, samples in patient_samples.items():
            is_train = patient_id in train_patients
            
            for sample_data in samples:
                if is_train:
                    train_samples_list.append(sample_data)
                else:
                    val_samples_list.append(sample_data)
        
    else:
        print(f"Random splitting (train ratio: {train_ratio})...")
        # 随机划分，不考虑病人ID
        train_samples_list, val_samples_list = split_by_sample_random(all_samples, train_ratio, seed)
        
        # 为每个样本生成唯一ID
        for idx, sample in enumerate(train_samples_list):
            ct_path = sample.get("ct_path", "")
            patient_id = extract_patient_id(ct_path) if ct_path else f"unknown_{idx}"
            sample["sample_id"] = f"patient_{patient_id}_sample_{idx:04d}_train"
            
        for idx, sample in enumerate(val_samples_list):
            ct_path = sample.get("ct_path", "")
            patient_id = extract_patient_id(ct_path) if ct_path else f"unknown_{idx+len(train_samples_list)}"
            sample["sample_id"] = f"patient_{patient_id}_sample_{idx+len(train_samples_list):04d}_val"
        
        print(f"Training samples: {len(train_samples_list)}")
        print(f"Validation samples: {len(val_samples_list)}")
        
        # 准备训练和验证文件内容
        train_lines = []
        val_lines = []
        
        total_objects = 0
        processed_train = 0
        processed_val = 0
    
    # 处理训练集样本
    for sample_data in train_samples_list:
        sample_id = sample_data["sample_id"]
        ct_path = sample_data["ct_path"]
        pet_path = sample_data.get("pet_path", "")
        mask_path = sample_data["mask_path"]
        
        print(f"Processing training sample: {sample_id}")
        
        # 转换CT图像为PNG
        ct_image_path = None
        if ct_path and os.path.exists(ct_path):
            ct_image_path = nii_to_image(ct_path, images_dir, sample_id, "ct")
        
        # 转换PET图像为PNG（如果启用）
        pet_image_path = None
        if process_pet and pet_path and os.path.exists(pet_path):
            pet_image_path = nii_to_image(pet_path, images_dir, sample_id, "pet")
        elif process_pet:
            # 如果启用PET处理但没有PET路径，使用CT路径占位
            pet_image_path = ct_image_path
        
        # 从mask生成XML标注
        xml_path, num_objects, bbox_list = None, 0, []
        if mask_path and os.path.exists(mask_path):
            xml_path, num_objects, bbox_list = mask_to_xml(
                mask_path, annotations_dir, sample_id, 
                image_size[0], image_size[1]
            )
        
        if num_objects > 0 and ct_image_path:
            total_objects += num_objects
            
            # 构建YOLO格式的标注行
            bbox_str_list = []
            for bbox in bbox_list:
                bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},0"
                bbox_str_list.append(bbox_str)
            
            # 生成训练文件行
            all_bboxes = " ".join(bbox_str_list)
            
            # 根据是否处理PET决定行格式
            if process_pet and pet_image_path:
                line = f"{ct_image_path} {pet_image_path} {all_bboxes}"
            else:
                # 如果不处理PET，只使用CT路径
                line = f"{ct_image_path} {all_bboxes}"
            
            train_lines.append(line)
            processed_train += 1
            
            print(f"  -> Generated annotations with {num_objects} objects")
    
    # 处理验证集样本
    for sample_data in val_samples_list:
        sample_id = sample_data["sample_id"]
        ct_path = sample_data["ct_path"]
        pet_path = sample_data.get("pet_path", "")
        mask_path = sample_data["mask_path"]
        
        print(f"Processing validation sample: {sample_id}")
        
        # 转换CT图像为PNG
        ct_image_path = None
        if ct_path and os.path.exists(ct_path):
            ct_image_path = nii_to_image(ct_path, images_dir, sample_id, "ct")
        
        # 转换PET图像为PNG（如果启用）
        pet_image_path = None
        if process_pet and pet_path and os.path.exists(pet_path):
            pet_image_path = nii_to_image(pet_path, images_dir, sample_id, "pet")
        elif process_pet:
            # 如果启用PET处理但没有PET路径，使用CT路径占位
            pet_image_path = ct_image_path
        
        # 从mask生成XML标注
        xml_path, num_objects, bbox_list = None, 0, []
        if mask_path and os.path.exists(mask_path):
            xml_path, num_objects, bbox_list = mask_to_xml(
                mask_path, annotations_dir, sample_id, 
                image_size[0], image_size[1]
            )
        
        if num_objects > 0 and ct_image_path:
            total_objects += num_objects
            
            # 构建YOLO格式的标注行
            bbox_str_list = []
            for bbox in bbox_list:
                bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},0"
                bbox_str_list.append(bbox_str)
            
            # 生成验证文件行
            all_bboxes = " ".join(bbox_str_list)
            
            # 根据是否处理PET决定行格式
            if process_pet and pet_image_path:
                line = f"{ct_image_path} {pet_image_path} {all_bboxes}"
            else:
                # 如果不处理PET，只使用CT路径
                line = f"{ct_image_path} {all_bboxes}"
            
            val_lines.append(line)
            processed_val += 1
            
            print(f"  -> Generated annotations with {num_objects} objects")
    
    # 保存训练和验证文件
    train_file = os.path.join(image_sets_dir, "2007_train.txt")
    val_file = os.path.join(image_sets_dir, "2007_val.txt")
    
    with open(train_file, 'w') as f:
        f.write("\n".join(train_lines))
    
    with open(val_file, 'w') as f:
        f.write("\n".join(val_lines))
    
    # 创建训练和验证的ID文件（只包含样本ID）
    train_id_file = os.path.join(image_sets_dir, "train.txt")
    val_id_file = os.path.join(image_sets_dir, "val.txt")
    
    # 提取样本ID
    train_ids = []
    for line in train_lines:
        # 根据行格式提取CT路径
        parts = line.split()
        if process_pet and len(parts) > 2:
            ct_path = parts[0]
        else:
            ct_path = parts[0]
        sample_id = os.path.basename(ct_path).replace("_ct.png", "")
        train_ids.append(sample_id)
    
    val_ids = []
    for line in val_lines:
        # 根据行格式提取CT路径
        parts = line.split()
        if process_pet and len(parts) > 2:
            ct_path = parts[0]
        else:
            ct_path = parts[0]
        sample_id = os.path.basename(ct_path).replace("_ct.png", "")
        val_ids.append(sample_id)
    
    with open(train_id_file, 'w') as f:
        f.write("\n".join(train_ids))
    
    with open(val_id_file, 'w') as f:
        f.write("\n".join(val_ids))
    
    # 创建类别文件
    classes_txt = os.path.join(output_base_dir, "classes.txt")
    with open(classes_txt, 'w') as f:
        f.write("lesion\n")
    
    # 创建数据集统计文件
    stats_file = os.path.join(output_base_dir, "dataset_stats.txt")
    with open(stats_file, 'w') as f:
        f.write(f"JSON file: {json_path}\n")
        f.write(f"Output directory: {output_base_dir}\n")
        f.write(f"Process PET: {process_pet}\n")
        f.write(f"Split by patient: {split_by_patient_flag}\n")
        
        if split_by_patient_flag:
            f.write(f"Splitting method: Fixed 63:8 patient ratio\n")
            f.write(f"Random seed: {seed}\n")
            f.write(f"Total unique patients: {len(patient_samples)}\n")
            f.write(f"Training patients: {len(train_patients)} (63)\n")
            f.write(f"Validation patients: {len(val_patients)} (8)\n")
        else:
            f.write(f"Splitting method: Random sample split\n")
            f.write(f"Train ratio: {train_ratio}\n")
            f.write(f"Random seed: {seed}\n")
        
        f.write(f"Training samples: {processed_train}\n")
        f.write(f"Validation samples: {processed_val}\n")
        f.write(f"Total objects detected: {total_objects}\n")
        f.write(f"Average objects per sample: {total_objects/(processed_train+processed_val):.2f}\n")
    
    print(f"\nConversion completed!")
    print(f"Process PET: {process_pet}")
    print(f"Split by patient: {split_by_patient_flag}")
    
    if split_by_patient_flag:
        print(f"Total unique patients: {len(patient_samples)}")
        print(f"Training patients: {len(train_patients)} (63), Validation patients: {len(val_patients)} (8)")
    
    print(f"Training samples: {processed_train}, Validation samples: {processed_val}")
    print(f"Total objects detected: {total_objects}")
    print(f"Average objects per sample: {total_objects/(processed_train+processed_val):.2f}")
    print(f"\nVOC directory structure created at: {output_base_dir}")
    print(f"Training file: {train_file}")
    print(f"Validation file: {val_file}")
    print(f"Training IDs: {train_id_file}")
    print(f"Validation IDs: {val_id_file}")
    print(f"Classes file: {classes_txt}")
    print(f"Dataset stats: {stats_file}")
    
    # 打印病人ID分配情况（如果按病人划分）
    if split_by_patient_flag:
        print("\nPatient ID assignment:")
        print(f"Training patients ({len(train_patients)}): {sorted(train_patients)}")
        print(f"Validation patients ({len(val_patients)}): {sorted(val_patients)}")

def main():
    """主函数，解析命令行参数"""
    parser = argparse.ArgumentParser(description='Convert medical imaging dataset to VOC format')
    
    parser.add_argument('--json', type=str, required=True,
                        help='Path to JSON file containing dataset information')
    
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for VOC format dataset')
    
    parser.add_argument('--image_size', type=int, nargs=2, default=[384, 384],
                        help='Image size (width height), default: 384 384')
    
    parser.add_argument('--process_pet', action='store_true',
                        help='Process PET images (default: False)')
    
    parser.add_argument('--no_process_pet', action='store_false', dest='process_pet',
                        help='Do not process PET images')
    
    parser.add_argument('--split_by_patient', action='store_true',
                        help='Split dataset by patient ID (default: True)')
    
    parser.add_argument('--no_split_by_patient', action='store_false', dest='split_by_patient',
                        help='Split dataset randomly (ignoring patient ID)')
    
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio (0-1), default: 0.8 (used only when not splitting by patient)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed, default: 42')
    
    parser.set_defaults(process_pet=False, split_by_patient=True)
    
    args = parser.parse_args()
    
    # 执行转换
    convert_to_voc(
        json_path=args.json,
        output_base_dir=args.output,
        image_size=tuple(args.image_size),
        process_pet=args.process_pet,
        split_by_patient_flag=args.split_by_patient,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

# 使用示例
if __name__ == "__main__":
    # 保留原有的调用方式
    json_file = "/root/autodl-tmp/LUNA16_datasplit/71_2d.json"
    output_dir = "/root/autodl-tmp/detection/model/yolox-pytorch-main/VOC_CT_Dataset_5"
    image_size = (384, 384)
    
    # 使用原有的调用方式（按病人划分，63:8，随机种子42）
    convert_to_voc(json_file, output_dir, image_size, 
                   process_pet=True, split_by_patient_flag=True, 
                   train_ratio=0.8, seed=42)
    
    # 也可以使用命令行参数
    # main()