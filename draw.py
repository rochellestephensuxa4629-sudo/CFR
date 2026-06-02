import os
from PIL import Image, ImageDraw, ImageFont

gt_dir = "/root/autodl-tmp/detection/model/yolox-pytorch-main/new_map_yiyuan_onlyct/ground-truth"
pred_dir = "/root/autodl-tmp/detection/model/yolox-pytorch-main/new_map_yiyuan_onlyct/detection-results"
img_dir = "/root/autodl-tmp/detection/model/yolox-pytorch-main/VOC_CT_Dataset_3_feichuang/JPEGImages"
output_dir = "/root/autodl-tmp/detection/model/yolox-pytorch-main/new_map_yiyuan_onlyct/visualized_results3"

conf_threshold = 0.5
target_size = (384, 384)
gt_color = "green"
pred_color = "red"
font_size = 16
draw_pet = False

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_gt_line(line):
    parts = line.strip().split()
    if len(parts) == 5:
        cls = parts[0]
        x_min, y_min, x_max, y_max = map(int, parts[1:5])
        return 'GTV', (x_min, y_min, x_max, y_max)
    return None

def parse_pred_line(line, threshold):
    parts = line.strip().split()
    if len(parts) == 6:
        cls = parts[0]
        conf = float(parts[1])
        if conf < threshold:
            return None
        x_min, y_min, x_max, y_max = map(int, parts[2:6])
        return cls, conf, (x_min, y_min, x_max, y_max)
    return None

def rotate_flip_bbox(bbox, orig_w, orig_h):
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    new_corners = []
    for x, y in corners:
        x_rot = y
        y_rot = orig_w - 1 - x
        x_final = orig_h - 1 - x_rot
        y_final = y_rot
        new_corners.append((x_final, y_final))
    new_x1 = min(c[0] for c in new_corners)
    new_y1 = min(c[1] for c in new_corners)
    new_x2 = max(c[0] for c in new_corners)
    new_y2 = max(c[1] for c in new_corners)
    return (384 - new_x1, 384 - new_y1, 384 - new_x2, 384 - new_y2)

def draw_boxes_only(image, bboxes, color):
    draw = ImageDraw.Draw(image)
    for cls, (x1, y1, x2, y2) in bboxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    return image

def draw_labels_on_image(image, labels_and_positions, color, font):
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    for label, (x, y) in labels_and_positions:
        y_text = y - font_size - 2
        if y_text < 0:
            y_text = 0
        text_bbox = draw.textbbox((x, y_text), label, font=font)
        if text_bbox[2] > img_w:
            x_adj = max(0, x - (text_bbox[2] - img_w))
            text_bbox = draw.textbbox((x_adj, y_text), label, font=font)
        else:
            x_adj = x
        draw.rectangle(text_bbox, fill=color)
        draw.text((x_adj, y_text), label, fill="white", font=font)
    return image

def process_sample(sample_name):
    gt_path = os.path.join(gt_dir, sample_name + ".txt")
    pred_path = os.path.join(pred_dir, sample_name + ".txt")
    ct_path = os.path.join(img_dir, sample_name + "_ct.png")
    pet_path = os.path.join(img_dir, sample_name + "_pet.png")

    if not os.path.exists(ct_path):
        print(f"警告：CT图像不存在 {ct_path}，跳过样本 {sample_name}")
        return

    pet_img = None
    if draw_pet:
        if not os.path.exists(pet_path):
            print(f"警告：PET图像不存在 {pet_path}，将只输出CT图像（跳过PET）")
            effective_draw_pet = False
        else:
            pet_img = Image.open(pet_path).convert("RGB")
            effective_draw_pet = True
    else:
        effective_draw_pet = False

    # 读取CT图像
    ct_img = Image.open(ct_path).convert("RGB")
    orig_w, orig_h = ct_img.size

    # ===== 新增：生成并保存旋转翻转后的无标注图像 =====
    ct_clean = ct_img.copy()                     # 原始图像副本
    ct_clean = ct_clean.transpose(Image.ROTATE_270)
    ct_clean = ct_clean.transpose(Image.FLIP_LEFT_RIGHT)
    clean_out_path = os.path.join(output_dir, f"{sample_name}_clean.png")
    ct_clean.save(clean_out_path)
    print(f"已保存无标注图像: {clean_out_path}")

    # 读取真实框
    gt_bboxes = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                parsed = parse_gt_line(line)
                if parsed:
                    gt_bboxes.append(parsed)
    else:
        print(f"警告：真实框文件不存在 {gt_path}，将只绘制预测框（如有）")

    # 读取预测框（已过滤置信度）
    pred_bboxes = []
    if os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            for line in f:
                parsed = parse_pred_line(line, conf_threshold)
                if parsed:
                    cls, conf, bbox = parsed
                    label = f"GTV {conf:.2f}"
                    pred_bboxes.append((label, bbox))
    else:
        print(f"警告：预测结果文件不存在 {pred_path}，将只绘制真实框")

    # 在原始图像上绘制框（无文字）
    ct_with_boxes = ct_img.copy()
    ct_with_boxes = draw_boxes_only(ct_with_boxes, gt_bboxes, color=gt_color)
    ct_with_boxes = draw_boxes_only(ct_with_boxes, pred_bboxes, color=pred_color)

    # 旋转翻转（与无标注图像保持一致）
    ct_with_boxes = ct_with_boxes.transpose(Image.ROTATE_270)
    ct_with_boxes = ct_with_boxes.transpose(Image.FLIP_LEFT_RIGHT)
    new_w, new_h = ct_with_boxes.size

    # 计算文字的新位置（在原图旋转后的坐标系中）
    gt_labels_pos = []
    for cls, bbox in gt_bboxes:
        new_bbox = rotate_flip_bbox(bbox, orig_w, orig_h)
        gt_labels_pos.append((cls, (new_bbox[2], new_bbox[3])))

    pred_labels_pos = []
    for label, bbox in pred_bboxes:
        new_bbox = rotate_flip_bbox(bbox, orig_w, orig_h)
        pred_labels_pos.append((label, (new_bbox[2], new_bbox[3])))

    # 设置字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # 在变换后的图像上绘制水平文字
    ct_with_boxes = draw_labels_on_image(ct_with_boxes, gt_labels_pos, color=gt_color, font=font)
    ct_with_boxes = draw_labels_on_image(ct_with_boxes, pred_labels_pos, color=pred_color, font=font)

    # 缩放至目标尺寸
    ct_resized = ct_with_boxes.resize(target_size, Image.Resampling.LANCZOS)

    # 保存或拼接（与之前逻辑一致）
    if effective_draw_pet and pet_img is not None:
        pet_resized = pet_img.resize(target_size, Image.Resampling.LANCZOS)
        total_width = ct_resized.width + pet_resized.width + 10
        max_height = max(ct_resized.height, pet_resized.height)
        combined = Image.new('RGB', (total_width, max_height), color='white')
        combined.paste(ct_resized, (0, 0))
        combined.paste(pet_resized, (ct_resized.width + 10, 0))
        draw = ImageDraw.Draw(combined)
        draw.text((5, 5), "CT (rotated+flipped) with bboxes", fill="black", font=font)
        draw.text((ct_resized.width + 15, 5), "PET", fill="black", font=font)
        out_path = os.path.join(output_dir, f"{sample_name}_visualized.png")
        combined.save(out_path)
        print(f"已保存（含PET）: {out_path}")
    else:
        out_path = os.path.join(output_dir, f"{sample_name}_visualized.png")
        ct_resized.save(out_path)
        print(f"已保存（仅CT）: {out_path}")

if __name__ == "__main__":
    ensure_dir(output_dir)
    if not os.path.exists(gt_dir):
        raise FileNotFoundError(f"真实框目录不存在：{gt_dir}")

    sample_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
    if not sample_files:
        print("未找到任何真实框文件，请检查路径。")
        exit()

    print(f"找到 {len(sample_files)} 个样本，开始处理...")
    for file_name in sample_files:
        sample_name = os.path.splitext(file_name)[0]
        process_sample(sample_name)

    print("所有样本处理完成！")