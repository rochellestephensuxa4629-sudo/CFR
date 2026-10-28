import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# 设置路径
project_root = os.path.dirname(os.path.abspath(__file__))
map_root = os.environ.get("MAP_OUT", os.path.join(project_root, "map_out"))
detection_results_dir = os.path.join(map_root, "detection-results")
ground_truth_dir = os.path.join(map_root, "ground-truth")
images_dir = os.environ.get("IMAGE_DIR", os.path.join(project_root, "data", "JPEGImages_CT"))
output_dir = os.environ.get("VISUALIZATION_DIR", os.path.join(project_root, "visualizations"))

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

def parse_detection_file(filepath, confidence_threshold=0.5):
    """解析检测结果文件，只返回置信度≥threshold的框"""
    boxes = []
    try:
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 6:  # 处理可能的额外空格
                    class_name = parts[0]
                    confidence = float(parts[1])
                    # 只保留置信度≥threshold的框
                    if confidence >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, parts[2:6])
                        boxes.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        })
                elif len(parts) == 5:  # 标准格式：class confidence x1 y1 x2 y2
                    class_name = parts[0]
                    confidence = float(parts[1])
                    if confidence >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, parts[2:])
                        boxes.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        })
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    
    return boxes

def parse_ground_truth_file(filepath):
    """解析ground truth文件"""
    boxes = []
    try:
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    class_name = parts[0]
                    x1, y1, x2, y2 = map(int, parts[1:5])
                    boxes.append({
                        'class': class_name,
                        'bbox': [x1, y1, x2, y2]
                    })
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    
    return boxes

def draw_boxes_on_image(image, pred_boxes, gt_boxes, confidence_threshold=0.5):
    """
    在图像上绘制预测框和真实框
    
    Args:
        image: 原始图像（numpy数组）
        pred_boxes: 预测框列表
        gt_boxes: 真实框列表
        confidence_threshold: 置信度阈值（只显示≥此值的预测框）
    """
    # 创建图像副本
    img_with_boxes = image.copy()
    
    # 如果图像是灰度图，转换为RGB
    if len(img_with_boxes.shape) == 2:
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_GRAY2RGB)
    elif len(img_with_boxes.shape) == 3 and img_with_boxes.shape[2] == 1:
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_GRAY2RGB)
    
    # 1. 先绘制真实框（绿色）
    for box in gt_boxes:
        x1, y1, x2, y2 = box['bbox']
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加标签（不含置信度）
        label = f"GT: {box['class']}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_with_boxes, (x1, y1 - label_size[1] - 5), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(img_with_boxes, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 2. 再绘制预测框（红色），只绘制置信度≥threshold的
    for box in pred_boxes:
        if box['confidence'] >= confidence_threshold:
            x1, y1, x2, y2 = box['bbox']
            # 绘制矩形框
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 添加标签（包含置信度）
            label = f"{box['class']}: {box['confidence']:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_with_boxes, (x1, y2), 
                         (x1 + label_size[0], y2 + label_size[1] + 5), (255, 0, 0), -1)
            cv2.putText(img_with_boxes, label, (x1, y2 + label_size[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_with_boxes

def visualize_single_sample(txt_filename, confidence_threshold=0.5):
    """可视化单个样本"""
    # 构建文件路径
    txt_name = os.path.splitext(txt_filename)[0]  # 去除.txt后缀
    pred_file = os.path.join(detection_results_dir, txt_filename)
    gt_file = os.path.join(ground_truth_dir, txt_filename)
    image_file = os.path.join(images_dir, f"{txt_name}_ct.png")
    
    # 检查文件是否存在
    if not os.path.exists(pred_file):
        print(f"检测结果文件不存在: {pred_file}")
        return None
    
    if not os.path.exists(gt_file):
        print(f"Ground truth文件不存在: {gt_file}")
        return None
    
    if not os.path.exists(image_file):
        # 尝试其他可能的图像格式
        image_file_jpg = os.path.join(images_dir, f"{txt_name}_ct.jpg")
        if os.path.exists(image_file_jpg):
            image_file = image_file_jpg
        else:
            print(f"图像文件不存在: {image_file}")
            return None
    
    # 解析文件
    pred_boxes = parse_detection_file(pred_file, confidence_threshold)
    gt_boxes = parse_ground_truth_file(gt_file)
    
    # 读取图像
    try:
        # 使用PIL读取（处理可能的各种格式）
        image = Image.open(image_file)
        image = np.array(image)
        
        # 如果是RGBA，转换为RGB
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    except Exception as e:
        print(f"读取图像失败 {image_file}: {e}")
        return None
    
    # 绘制框
    result_image = draw_boxes_on_image(image, pred_boxes, gt_boxes, confidence_threshold)
    
    # 统计信息
    num_pred = len([b for b in pred_boxes if b['confidence'] >= confidence_threshold])
    num_gt = len(gt_boxes)
    
    # 在图像上添加统计信息
    stats_text = f"Predictions (≥{confidence_threshold}): {num_pred}, Ground Truth: {num_gt}"
    cv2.putText(result_image, stats_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 保存结果
    output_path = os.path.join(output_dir, f"{txt_name}_visualization.png")
    cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
    return {
        'pred_boxes': pred_boxes,
        'gt_boxes': gt_boxes,
        'image_path': output_path,
        'num_pred': num_pred,
        'num_gt': num_gt
    }

def visualize_all_samples(confidence_threshold=0.5, max_samples=None):
    """可视化所有样本"""
    # 获取所有检测结果文件
    all_txt_files = [f for f in os.listdir(detection_results_dir) if f.endswith('.txt')]
    
    if max_samples:
        all_txt_files = all_txt_files[:max_samples]
    
    print(f"找到 {len(all_txt_files)} 个检测结果文件")
    
    results_summary = []
    
    for i, txt_file in enumerate(all_txt_files):
        print(f"处理 {i+1}/{len(all_txt_files)}: {txt_file}")
        
        result = visualize_single_sample(txt_file, confidence_threshold)
        
        if result:
            results_summary.append({
                'filename': txt_file,
                'num_pred': result['num_pred'],
                'num_gt': result['num_gt']
            })
    
    # 生成汇总报告
    if results_summary:
        total_pred = sum(r['num_pred'] for r in results_summary)
        total_gt = sum(r['num_gt'] for r in results_summary)
        total_files = len(results_summary)
        
        print("\n" + "="*50)
        print("可视化完成！")
        print(f"总处理文件数: {total_files}")
        print(f"总预测框数（置信度≥{confidence_threshold}）: {total_pred}")
        print(f"总真实框数: {total_gt}")
        print(f"平均每张图预测框数: {total_pred/total_files:.2f}")
        print(f"平均每张图真实框数: {total_gt/total_files:.2f}")
        print(f"结果保存在: {output_dir}")
        
        # 保存汇总报告
        report_path = os.path.join(output_dir, "visualization_report.txt")
        with open(report_path, 'w') as f:
            f.write("可视化结果汇总报告\n")
            f.write("="*50 + "\n")
            f.write(f"置信度阈值: {confidence_threshold}\n")
            f.write(f"总处理文件数: {total_files}\n")
            f.write(f"总预测框数: {total_pred}\n")
            f.write(f"总真实框数: {total_gt}\n")
            f.write(f"平均每张图预测框数: {total_pred/total_files:.2f}\n")
            f.write(f"平均每张图真实框数: {total_gt/total_files:.2f}\n\n")
            f.write("各文件详情:\n")
            for r in results_summary:
                f.write(f"{r['filename']}: 预测框={r['num_pred']}, 真实框={r['num_gt']}\n")
        
        print(f"详细报告保存在: {report_path}")

def create_comparison_visualization(confidence_thresholds=[0.1, 0.3, 0.5, 0.7]):
    """创建不同置信度阈值的对比可视化"""
    # 随机选择一个样本来展示不同阈值的效果
    import random
    
    all_txt_files = [f for f in os.listdir(detection_results_dir) if f.endswith('.txt')]
    if not all_txt_files:
        print("没有找到检测结果文件")
        return
    
    sample_file = random.choice(all_txt_files)
    print(f"为阈值对比选择的样本: {sample_file}")
    
    # 读取图像
    txt_name = os.path.splitext(sample_file)[0]
    image_file = os.path.join(images_dir, f"{txt_name}_ct.png")
    if not os.path.exists(image_file):
        return
    
    try:
        image = Image.open(image_file)
        image = np.array(image)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    except:
        return
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, threshold in enumerate(confidence_thresholds):
        # 解析预测框和真实框
        pred_file = os.path.join(detection_results_dir, sample_file)
        gt_file = os.path.join(ground_truth_dir, sample_file)
        
        pred_boxes = parse_detection_file(pred_file, threshold)
        gt_boxes = parse_ground_truth_file(gt_file)
        
        # 绘制图像
        result_image = draw_boxes_on_image(image.copy(), pred_boxes, gt_boxes, threshold)
        
        # 显示
        axes[idx].imshow(result_image)
        axes[idx].set_title(f"Confidence Threshold = {threshold}\n"
                           f"Predictions: {len([b for b in pred_boxes if b['confidence'] >= threshold])}, "
                           f"Ground Truth: {len(gt_boxes)}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, f"{txt_name}_threshold_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"阈值对比图已保存: {comparison_path}")

if __name__ == "__main__":
    # 1. 可视化所有样本（置信度阈值设为0.5）
    print("开始可视化所有样本...")
    visualize_all_samples(confidence_threshold=0.5)
    
    # 2. 创建阈值对比图（可选）
    print("\n创建阈值对比图...")
    create_comparison_visualization()
    
    # 3. 你也可以单独可视化某个样本
    # result = visualize_single_sample("patient_140701512_sample_0560.txt", confidence_threshold=0.5)
    # if result:
    #     print(f"已保存: {result['image_path']}")
