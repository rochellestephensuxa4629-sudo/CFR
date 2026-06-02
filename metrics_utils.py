#!/usr/bin/env python
# coding: utf-8
"""
评估指标计算工具函数
包含：精确率、召回率、F1、AP@50、AP@50-95、ROC-AUC、可视化检测图
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import torch
from torchvision.ops import box_iou, nms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---------- Soft-NMS 实现 ----------
def soft_nms_pytorch(boxes, scores, sigma=0.5, score_thresh=0.001, method='gaussian'):
    """Soft-NMS实现"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    boxes_np = boxes.cpu().numpy().astype(float)
    scores_np = scores.cpu().numpy().astype(float)
    N = boxes_np.shape[0]
    indexes = np.arange(N)
    keep = []

    while indexes.size > 0:
        max_idx = np.argmax(scores_np[indexes])
        curr = indexes[max_idx]
        keep.append(curr)
        indexes = np.delete(indexes, max_idx)
        if indexes.size == 0:
            break

        ious = []
        x1 = boxes_np[indexes, 0]
        y1 = boxes_np[indexes, 1]
        x2 = boxes_np[indexes, 2]
        y2 = boxes_np[indexes, 3]

        xx1 = np.maximum(boxes_np[curr, 0], x1)
        yy1 = np.maximum(boxes_np[curr, 1], y1)
        xx2 = np.minimum(boxes_np[curr, 2], x2)
        yy2 = np.minimum(boxes_np[curr, 3], y2)

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_curr = (boxes_np[curr, 2] - boxes_np[curr, 0]) * (boxes_np[curr, 3] - boxes_np[curr, 1])
        area_others = (x2 - x1) * (y2 - y1)
        union = area_curr + area_others - inter
        iou_vals = inter / (union + 1e-6)

        if method == 'linear':
            weight = np.ones_like(iou_vals)
            weight[iou_vals > 0.5] = 1 - iou_vals[iou_vals > 0.5]
        elif method == 'gaussian':
            weight = np.exp(-(iou_vals ** 2) / sigma)
        else:
            weight = np.ones_like(iou_vals)
            weight[iou_vals > 0.5] = 1 - iou_vals[iou_vals > 0.5]

        scores_np[indexes] = scores_np[indexes] * weight

        remain_mask = scores_np[indexes] >= score_thresh
        indexes = indexes[remain_mask]

    keep = np.array(keep, dtype=int)
    return torch.from_numpy(keep).long()


# ---------- NMS应用函数 ----------
def apply_nms_to_prediction(pred, score_thresh=0.05, nms_iou=0.5, use_soft_nms=False, soft_nms_sigma=0.5, soft_nms_score_thresh=0.05):
    """对单张预测结果应用score threshold和NMS"""
    if len(pred.get("boxes", [])) == 0:
        return {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.int64)}

    boxes = pred["boxes"]
    scores = pred["scores"]
    labels = pred.get("labels", torch.zeros_like(scores, dtype=torch.int64))

    # 1. 置信度过滤
    keep_score = scores >= score_thresh
    if keep_score.sum() == 0:
        return {"boxes": torch.zeros((0, 4)), "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.int64)}

    boxes_f = boxes[keep_score]
    scores_f = scores[keep_score]
    labels_f = labels[keep_score]

    # 2. Soft-NMS 或标准 NMS 去重
    try:
        if use_soft_nms:
            keep = soft_nms_pytorch(boxes_f, scores_f, sigma=soft_nms_sigma, 
                                    score_thresh=soft_nms_score_thresh, method='gaussian')
        else:
            keep = nms(boxes_f, scores_f, nms_iou)
    except Exception:
        keep = torch.arange(len(scores_f))

    boxes_out = boxes_f[keep]
    scores_out = scores_f[keep]
    labels_out = labels_f[keep] if labels_f is not None else torch.zeros_like(scores_out, dtype=torch.int64)

    return {"boxes": boxes_out, "scores": scores_out, "labels": labels_out}


# ---------- ROC-AUC计算函数 ----------
def calculate_roc_auc_interpolation(model, data_loader, device, iou_thresh=0.5):
    """
    目标级别ROC-AUC计算（插值法）
    返回: roc_auc, fpr, tpr
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (imgs, targets, _) in enumerate(data_loader):
            imgs_dev = [img.to(device) for img in imgs]
            preds = model(imgs_dev)

            for pred, target in zip(preds, targets):
                gt_boxes = target['boxes'].cpu()
                pred_boxes = pred['boxes'].cpu()
                pred_scores = pred['scores'].cpu()

                if len(gt_boxes) == 0:
                    # 无GT：所有预测框为负样本
                    all_scores.extend(pred_scores.tolist())
                    all_labels.extend([0] * len(pred_scores))
                    continue

                if len(pred_boxes) == 0:
                    # 无预测：无样本
                    continue

                # 计算IoU矩阵
                ious = box_iou(pred_boxes, gt_boxes)
                max_ious, max_gt_idx = ious.max(dim=1)

                # 标记已匹配的GT
                matched_gt = set()
                for i, (iou, score) in enumerate(zip(max_ious, pred_scores)):
                    if iou >= iou_thresh and max_gt_idx[i] not in matched_gt:
                        all_scores.append(score.item())
                        all_labels.append(1)
                        matched_gt.add(max_gt_idx[i])
                    else:
                        all_scores.append(score.item())
                        all_labels.append(0)

    if len(all_scores) == 0:
        print("[WARNING] 没有有效的预测分数用于计算ROC-AUC")
        return 0.0, [], []

    scores_array = np.array(all_scores)
    labels_array = np.array(all_labels)
    
    # 使用sklearn的标准方法计算ROC-AUC（内置插值）
    fpr, tpr, _ = roc_curve(labels_array, scores_array)
    roc_auc = auc(fpr, tpr)
    
    return roc_auc, fpr.tolist(), tpr.tolist()


# ---------- COCO评估函数 ----------
def evaluate_with_coco(model, data_loader, device, json_file, score_thresh=0.05, 
                      use_soft_nms=False, soft_nms_sigma=0.5, soft_nms_score_thresh=0.05):
    """
    使用COCO评估器计算指标，同时精准统计指定阈值下的Precision/Recall/F1
    返回: precision, recall, f1, map50, map5095, ar
    """
    model.eval()
    
    # 加载原始COCO数据
    with open(json_file, 'r', encoding='utf-8') as f:
        coco_gt_data = json.load(f)
    
    # 创建COCO ground truth对象
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_data
    coco_gt.createIndex()
    
    # 初始化变量
    results = []  # 用于COCOeval的预测结果
    image_ids = []
    # TP/FP/FN统计
    TP = 0
    FP = 0
    FN = 0
    all_scores = []
    all_tp_flags = []
    all_fp_flags = []

    with torch.no_grad():
        for batch_idx, (imgs, targets, img_ids) in enumerate(data_loader):
            imgs_dev = [img.to(device) for img in imgs]
            preds = model(imgs_dev)  # 仅一次前向推理！
            
            for pred, target, img_id in zip(preds, targets, img_ids):
                img_id_int = int(img_id)
                image_ids.append(img_id_int)
                
                # 1. 处理预测结果（NMS+阈值过滤）
                pred_cpu = {k: v.cpu() for k, v in pred.items()}
                pred_filtered = apply_nms_to_prediction(
                    pred_cpu, 
                    score_thresh=score_thresh, 
                    nms_iou=0.5,
                    use_soft_nms=use_soft_nms,
                    soft_nms_sigma=soft_nms_sigma,
                    soft_nms_score_thresh=soft_nms_score_thresh
                )
                pred_boxes = pred_filtered["boxes"]
                pred_scores = pred_filtered["scores"]
                pred_labels = pred_filtered["labels"]
                
                # 2. 收集COCO格式结果（用于COCOeval）
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    x1, y1, x2, y2 = box.tolist()
                    width = x2 - x1
                    height = y2 - y1
                    results.append({
                        "image_id": img_id_int,
                        "category_id": int(label),
                        "bbox": [x1, y1, width, height],
                        "score": float(score)
                    })
                
                # 3. 统计TP/FP/FN
                gt_boxes = target['boxes'].cpu()  # 当前图片的GT框
                num_gt = len(gt_boxes)
                
                if len(pred_boxes) == 0:
                    # 无预测结果 → 所有GT都是FN
                    FN += num_gt
                    continue
                
                if num_gt == 0:
                    # 无GT → 所有预测都是FP
                    FP += len(pred_boxes)
                    all_scores.extend(pred_scores.tolist())
                    all_tp_flags.extend([False] * len(pred_boxes))
                    all_fp_flags.extend([True] * len(pred_boxes))
                    continue
                
                # 计算IoU矩阵（预测框 vs GT框）
                ious = box_iou(pred_boxes, gt_boxes)  # [num_pred, num_gt]
                max_ious, max_gt_indices = ious.max(dim=1)  # 每个预测框匹配的最大IoU和GT索引
                
                # 核心：标记哪些GT被匹配（避免重复匹配）
                matched_gt_indices = set()
                true_positives = 0
                false_positives = 0
                
                for i in range(len(pred_boxes)):
                    iou_val = max_ious[i].item()
                    gt_idx = max_gt_indices[i].item()
                    
                    if iou_val >= 0.5 and gt_idx not in matched_gt_indices:
                        # 真正例：IoU≥0.5且GT未被匹配过
                        true_positives += 1
                        matched_gt_indices.add(gt_idx)
                        all_tp_flags.append(True)
                        all_fp_flags.append(False)
                    else:
                        # 假正例：IoU<0.5 或 GT已被匹配
                        false_positives += 1
                        all_tp_flags.append(False)
                        all_fp_flags.append(True)
                
                # 更新全局统计
                TP += true_positives
                FP += false_positives
                FN += (num_gt - len(matched_gt_indices))  # 未被匹配的GT是FN
                all_scores.extend(pred_scores.tolist())

    # 如果没有预测结果，返回默认值
    if len(results) == 0:
        print("[WARNING] 没有预测结果，返回默认评估值")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # ========== COCOeval计算mAP/AR ==========
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # 设置COCO评估参数（对齐行业标准）
    coco_eval.params.imgIds = image_ids
    coco_eval.params.catIds = [1]  # 单类别
    coco_eval.params.maxDets = [1, 10, 100]
    coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)  # 0.5到0.95，步长0.05（共10个）
    
    # 执行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 提取COCO指标
    map5095 = coco_eval.stats[0]  # mAP@0.5:0.95
    map50 = coco_eval.stats[1]    # mAP@0.5
    ar = coco_eval.stats[8]       # AR@100
    
    # ========== 计算指定阈值下的Precision/Recall/F1 ==========
    # 加1e-6避免除0
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    # 打印结果
    print(f"[COCO评估] mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map5095:.4f}, AR: {ar:.4f}")
    print(f"[指定阈值{score_thresh}] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return precision, recall, f1, map50, map5095, ar


# ---------- 验证函数 ----------
def validate_one_epoch(model, data_loader, device, json_file, score_thresh=0.05,
                      use_soft_nms=False, soft_nms_sigma=0.5, soft_nms_score_thresh=0.05):
    """验证函数，使用COCO评估器"""
    # 使用COCO评估器
    precision, recall, f1, map50, map5095, ar = evaluate_with_coco(
        model, data_loader, device, json_file, score_thresh,
        use_soft_nms, soft_nms_sigma, soft_nms_score_thresh
    )
    
    # 使用插值法计算ROC-AUC
    roc_auc, fpr_curve, tpr_curve = calculate_roc_auc_interpolation(model, data_loader, device, iou_thresh=0.5)
    
    print(f"Validation: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, "
          f"mAP@50={map50:.4f}, mAP@50:95={map5095:.4f}, AR={ar:.4f}, ROC-AUC={roc_auc:.4f}")
    
    return precision, recall, f1, map50, map5095, ar, roc_auc


# ---------- 阈值搜索函数 ----------
def find_best_threshold_on_validation(model, val_loader, device, json_file, iou_thresh=0.5,
                                     use_soft_nms=False, soft_nms_sigma=0.5, soft_nms_score_thresh=0.05):
    """在验证集上搜索最佳阈值，使用COCO评估器"""
    print("[INFO] 在验证集上搜索最佳阈值...")
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_f1 = 0.0
    best_thresh = 0.3
    best_metrics = {}

    for thresh in thresholds:
        print(f"  测试阈值={thresh:.2f}", end="\r")
        precision, recall, f1, map50, map5095, ar, _ = validate_one_epoch(
            model, val_loader, device, json_file, score_thresh=thresh,
            use_soft_nms=use_soft_nms, soft_nms_sigma=soft_nms_sigma,
            soft_nms_score_thresh=soft_nms_score_thresh
        )
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'map50': map50,
                'map5095': map5095,
                'ar': ar
            }
    
    print(f"\n[INFO] 验证集最佳阈值={best_thresh:.2f}")
    print(f"  Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}, "
          f"F1={best_metrics['f1']:.4f}, mAP@50={best_metrics['map50']:.4f}, "
          f"mAP@50:95={best_metrics['map5095']:.4f}, AR={best_metrics['ar']:.4f}")
    
    # 计算最佳阈值下的ROC-AUC
    roc_auc, fpr_curve, tpr_curve = calculate_roc_auc_interpolation(
        model, val_loader, device, iou_thresh
    )
    best_metrics['roc_auc'] = roc_auc
    
    return best_thresh, best_metrics


# ---------- ROC曲线绘制函数 ----------
def plot_roc_curve(fpr, tpr, roc_auc, save_path='./roc_curve.png', title_suffix=''):
    """绘制ROC曲线并保存"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve {title_suffix}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[INFO] ROC曲线已保存至: {save_path}")


# ---------- 可视化检测结果 ----------
def visualize_detections(test_set, predictions, img_indices, num_samples=4, 
                        save_dir='./detection_results', show_scores=False):
    """可视化检测结果"""
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(num_samples, len(test_set))
    idx_to_pred_idx = {}
    for pred_idx, original_idx in enumerate(img_indices):
        idx_to_pred_idx[original_idx] = pred_idx

    for i in range(num_samples):
        image, target, original_idx = test_set[i]
        pred_idx = idx_to_pred_idx.get(original_idx, i)
        if pred_idx >= len(predictions):
            print(f"[WARN] 预测索引 {pred_idx} 超出范围，跳过样本 {original_idx}")
            continue
        pred = predictions[pred_idx]
        image_np = image[:3].permute(1, 2, 0).cpu().numpy()
        image_np = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype(np.uint8)
        if image_np.shape[-1] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)
        
        # 绘制真实框（红色）
        if 'boxes' in target and len(target['boxes']) > 0:
            for box, label in zip(target['boxes'], target['labels']):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, color='red', linewidth=2,
                                     label='Ground Truth' if label == target['labels'][0] else "")
                plt.gca().add_patch(rect)
                # 只显示类别标签，不显示分数
                class_name = f"GT: {label.item()}"
                plt.text(x1, y1 - 5, class_name, color='red', fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.7))
        
        # 绘制预测框（绿色）
        if 'boxes' in pred and len(pred['boxes']) > 0:
            pred_boxes = pred['boxes'].cpu().numpy().astype(int)
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred.get('scores', torch.ones(len(pred_boxes))).cpu().numpy()
            
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     fill=False, color='green', linewidth=2, linestyle='--',
                                     label='Prediction' if label == pred_labels[0] else "")
                plt.gca().add_patch(rect)
                
                if show_scores:
                    # 显示分数和类别
                    display_text = f"Pred: {label} ({score:.2f})"
                else:
                    # 只显示类别，不显示分数
                    display_text = f"Pred: {label}"
                    
                plt.text(x1, y2 + 15, display_text, color='green', fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title(f"Detection Results - Sample {original_idx}")
        if ('boxes' in target and len(target['boxes']) > 0) or ('boxes' in pred and len(pred['boxes']) > 0):
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
        plt.axis('off')
        save_path = os.path.join(save_dir, f'detection_with_gt_{original_idx}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    print(f"[INFO] 检测结果已保存至 {save_dir}")


# ---------- 诊断函数 ----------
def debug_predictions(model, data_loader, device, num_samples=5):
    """调试预测结果，检查置信度分布"""
    model.eval()
    all_scores = []

    with torch.no_grad():
        for batch_idx, (imgs, targets, img_ids) in enumerate(data_loader):
            if batch_idx >= num_samples:
                break

            imgs_dev = [img.to(device) for img in imgs]
            preds = model(imgs_dev)

            for i, (pred, target, img_id) in enumerate(zip(preds, targets, img_ids)):
                print(f"\n--- 样本 {img_id} ---")
                print(f"预测框数量: {len(pred['boxes'])}")

                if len(pred['boxes']) > 0:
                    scores = pred['scores'].cpu().numpy()
                    print(f"预测分数: {scores}")
                    print(f"预测分数范围: {scores.min():.3f} - {scores.max():.3f}")
                    print(f"平均分数: {scores.mean():.3f}")
                    all_scores.extend(scores.tolist())

                print(f"真实框数量: {len(target['boxes'])}")
                if len(target['boxes']) > 0:
                    print(f"真实标签: {target['labels'].cpu().numpy()}")

    if all_scores:
        all_scores = np.array(all_scores)
        print(f"\n=== 置信度统计 ===")
        print(f"总预测框数: {len(all_scores)}")
        print(f"置信度范围: {all_scores.min():.3f} - {all_scores.max():.3f}")
        print(f"平均置信度: {all_scores.mean():.3f}")
        print(f"中位数置信度: {np.median(all_scores):.3f}")
        print(f">0.9的框数: {(all_scores > 0.9).sum()}")
        print(f">0.95的框数: {(all_scores > 0.95).sum()}")
        print(f"=1.00的框数: {(all_scores == 1.0).sum()}")


# ---------- 设置随机种子 ----------
def set_seed(seed=20):
    """设置随机种子"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

