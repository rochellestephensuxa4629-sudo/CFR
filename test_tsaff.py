#!/usr/bin/env python3
"""Evaluate a trained dual-modality detector and explain validation misses."""

import argparse
import csv
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_outputs, non_max_suppression


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a trained TSAFF/YOLOX model on the fixed validation split.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "logs" / "best_epoch_weights.pth"),
    )
    parser.add_argument(
        "--val-list",
        default=str(DATASET_ROOT / "ImageSets" / "Main" / "2007_val.txt"),
    )
    parser.add_argument(
        "--classes",
        default=str(DATASET_ROOT / "classes.txt"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "logs" / "tsaff_test"),
    )
    parser.add_argument("--model-module", default="nets.yolo_tsaff")
    parser.add_argument("--phi", default="l")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--tsaff-loops", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--operating-threshold", type=float, default=0.5)
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,"
        "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-partial-load", action="store_true")
    parser.add_argument(
        "--keep-float-boxes",
        action="store_true",
        help="Do not cast predicted boxes to integers before matching.",
    )
    parser.add_argument(
        "--keep-float-scores",
        action="store_true",
        help="Do not truncate scores to six characters before matching.",
    )
    return parser.parse_args()


def resolve_device(value):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(module_name, num_classes, phi, tsaff_loops, device):
    module = importlib.import_module(module_name)
    if not hasattr(module, "YoloBody"):
        raise AttributeError(f"{module_name} does not define YoloBody")

    body = module.YoloBody
    try:
        model = body(num_classes, phi, tsaff_loops=tsaff_loops)
    except TypeError:
        model = body(num_classes, phi)
    return model.to(device)


def load_checkpoint(model, checkpoint, device, allow_partial):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {
        key.replace("module.", "", 1): value
        for key, value in state.items()
    }

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing or unexpected:
        if not allow_partial:
            preview = list(missing[:10]) + list(unexpected[:10])
            raise RuntimeError(
                "Checkpoint does not match the model. "
                f"Use --allow-partial-load only for a deliberate partial load. {preview}"
            )


def parse_ground_truth(parts, start_index=2):
    boxes = []
    class_names = []
    for value in parts[start_index:]:
        values = value.split(",")
        if len(values) < 5:
            continue
        boxes.append([float(item) for item in values[:4]])
        class_names.append(int(values[4]))
    return boxes, class_names


def iou_xyxy(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def nms_box_to_xyxy(box):
    """Convert the project's NMS output order to left, top, right, bottom."""
    top, left, bottom, right = box
    return [float(left), float(top), float(right), float(bottom)]


def match_at_threshold(ground_truth, predictions, threshold, iou_threshold):
    used_ground_truth = set()
    matched = [False] * len(ground_truth)
    false_positives = 0

    ordered = sorted(
        predictions,
        key=lambda item: item["score"],
        reverse=True,
    )
    for prediction in ordered:
        if prediction["score"] < threshold:
            continue

        best_index = None
        best_iou = 0.0
        for index, target in enumerate(ground_truth):
            if index in used_ground_truth:
                continue
            if prediction["class_id"] != target["class_id"]:
                continue
            overlap = iou_xyxy(prediction["box"], target["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            matched[best_index] = True
            used_ground_truth.add(best_index)
        else:
            false_positives += 1

    return matched, false_positives


def calculate_ap(records, num_ground_truth, iou_threshold):
    if num_ground_truth == 0:
        return 0.0

    detections = []
    for image_index, record in enumerate(records):
        for prediction in record["predictions"]:
            detections.append((prediction["score"], image_index, prediction))
    detections.sort(key=lambda item: item[0], reverse=True)

    used_ground_truth = {}
    true_positive_flags = []
    false_positive_flags = []

    for _, image_index, prediction in detections:
        used = used_ground_truth.setdefault(image_index, set())
        ground_truth = records[image_index]["ground_truth"]
        best_index = None
        best_iou = 0.0

        for index, target in enumerate(ground_truth):
            if index in used:
                continue
            if prediction["class_id"] != target["class_id"]:
                continue
            overlap = iou_xyxy(prediction["box"], target["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            used.add(best_index)
            true_positive_flags.append(1.0)
            false_positive_flags.append(0.0)
        else:
            true_positive_flags.append(0.0)
            false_positive_flags.append(1.0)

    if not true_positive_flags:
        return 0.0

    true_positive = np.cumsum(true_positive_flags)
    false_positive = np.cumsum(false_positive_flags)
    recall = true_positive / max(num_ground_truth, 1)
    precision = true_positive / np.maximum(
        true_positive + false_positive,
        1e-12,
    )

    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(
        np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1])
    )


def image_id_from_path(path):
    stem = Path(path).stem
    if stem.endswith("_ct"):
        return stem[:-3]
    if stem.endswith("_pet"):
        return stem[:-4]
    return stem


def prepare_input(ct_image, pet_image, input_size):
    ct_image = cvtColor(ct_image)
    pet_image = cvtColor(pet_image)
    ct_data = resize_image(
        ct_image,
        (input_size, input_size),
        True,
    )
    pet_data = resize_image(
        pet_image,
        (input_size, input_size),
        True,
    )
    ct_data = np.transpose(
        preprocess_input(np.asarray(ct_data, dtype=np.float32)),
        (2, 0, 1),
    )
    pet_data = np.transpose(
        preprocess_input(np.asarray(pet_data, dtype=np.float32)),
        (2, 0, 1),
    )
    return ct_data, pet_data


def run_inference(model, samples, args, device, class_names):
    records = []
    num_ground_truth = 0

    with torch.no_grad():
        for start in tqdm(
            range(0, len(samples), args.batch_size),
            desc="Inference",
        ):
            batch = samples[start:start + args.batch_size]
            ct_batch = []
            pet_batch = []
            for sample in batch:
                ct_data, pet_data = prepare_input(
                    sample["ct_image"],
                    sample["pet_image"],
                    args.input_size,
                )
                ct_batch.append(ct_data)
                pet_batch.append(pet_data)

            ct_tensor = torch.from_numpy(np.asarray(ct_batch)).float().to(device)
            pet_tensor = torch.from_numpy(np.asarray(pet_batch)).float().to(device)
            outputs = model(ct_tensor, pet_tensor)
            decoded = decode_outputs(outputs, [args.input_size, args.input_size])
            detections = non_max_suppression(
                decoded,
                len(class_names),
                [args.input_size, args.input_size],
                np.asarray(sample["image_shape"]),
                True,
                conf_thres=args.confidence,
                nms_thres=args.nms_iou,
            )

            for sample, detection in zip(batch, detections):
                predictions = []
                if detection is not None:
                    for row in detection:
                        box = nms_box_to_xyxy(row[:4])
                        score = float(row[4] * row[5])
                        if not args.keep_float_boxes:
                            box = [int(value) for value in box]
                        if not args.keep_float_scores:
                            score = float(str(score)[:6])
                        predictions.append(
                            {
                                "box": box,
                                "score": score,
                                "class_id": int(row[6]),
                            }
                        )
                records.append(
                    {
                        "image_id": sample["image_id"],
                        "ground_truth": sample["ground_truth"],
                        "predictions": predictions,
                    }
                )
                num_ground_truth += len(sample["ground_truth"])

    return records, num_ground_truth


def evaluate_threshold(records, threshold, iou_threshold):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for record in records:
        matched, image_false_positives = match_at_threshold(
            record["ground_truth"],
            record["predictions"],
            threshold,
            iou_threshold,
        )
        true_positives += sum(matched)
        false_positives += image_false_positives
        false_negatives += len(matched) - sum(matched)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": threshold,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_lesion_rows(records, class_names, threshold, iou_threshold):
    rows = []
    for record in records:
        predictions = sorted(
            record["predictions"],
            key=lambda item: item["score"],
            reverse=True,
        )
        matched, _ = match_at_threshold(
            record["ground_truth"],
            predictions,
            threshold,
            iou_threshold,
        )

        for index, target in enumerate(record["ground_truth"]):
            best_same_class = None
            best_any_class = None
            for prediction in predictions:
                overlap = iou_xyxy(prediction["box"], target["box"])
                candidate = dict(prediction)
                candidate["iou"] = overlap
                if best_any_class is None or overlap > best_any_class["iou"]:
                    best_any_class = candidate
                if prediction["class_id"] == target["class_id"]:
                    if best_same_class is None or overlap > best_same_class["iou"]:
                        best_same_class = candidate

            if matched[index]:
                reason = "matched"
            elif best_same_class is None:
                reason = "no_prediction"
            elif best_same_class["iou"] < iou_threshold:
                reason = "localization_error"
            elif best_same_class["score"] < threshold:
                reason = "score_below_threshold"
            else:
                reason = "unmatched"

            rows.append(
                {
                    "image_id": record["image_id"],
                    "class": class_names[target["class_id"]],
                    "gt_box": " ".join(
                        f"{value:.2f}" for value in target["box"]
                    ),
                    "matched": int(matched[index]),
                    "reason": reason,
                    "best_score": (
                        f"{best_same_class['score']:.6f}"
                        if best_same_class is not None
                        else ""
                    ),
                    "best_iou": (
                        f"{best_same_class['iou']:.6f}"
                        if best_same_class is not None
                        else ""
                    ),
                    "best_pred_box": (
                        " ".join(
                            f"{value:.2f}"
                            for value in best_same_class["box"]
                        )
                        if best_same_class is not None
                        else ""
                    ),
                    "best_any_class": (
                        class_names[best_any_class["class_id"]]
                        if best_any_class is not None
                        else ""
                    ),
                }
            )
    return rows


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_thresholds(value):
    return sorted({float(item) for item in value.split(",") if item.strip()})


def main():
    args = parse_args()
    device = resolve_device(args.device)
    class_names, num_classes = get_classes(args.classes)
    thresholds = parse_thresholds(args.thresholds)

    if not thresholds:
        raise ValueError("At least one threshold is required.")
    if args.operating_threshold not in thresholds:
        thresholds.append(args.operating_threshold)
        thresholds.sort()

    samples = []
    with Path(args.val_list).open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            ct_image = Image.open(parts[0]).convert("RGB")
            pet_image = Image.open(parts[1]).convert("RGB")
            boxes, class_ids = parse_ground_truth(parts)
            ground_truth = [
                {"box": box, "class_id": class_id}
                for box, class_id in zip(boxes, class_ids)
            ]
            samples.append(
                {
                    "image_id": image_id_from_path(parts[0]),
                    "image_shape": [ct_image.size[1], ct_image.size[0]],
                    "ct_image": ct_image,
                    "pet_image": pet_image,
                    "ground_truth": ground_truth,
                }
            )

    model = build_model(
        args.model_module,
        num_classes,
        args.phi,
        args.tsaff_loops,
        device,
    )
    load_checkpoint(
        model,
        args.checkpoint,
        device,
        args.allow_partial_load,
    )
    model.eval()

    records, num_ground_truth = run_inference(
        model,
        samples,
        args,
        device,
        class_names,
    )

    threshold_rows = [
        evaluate_threshold(records, threshold, args.iou_threshold)
        for threshold in thresholds
    ]
    ap50 = calculate_ap(
        records,
        num_ground_truth,
        args.iou_threshold,
    )
    lesion_rows = build_lesion_rows(
        records,
        class_names,
        args.operating_threshold,
        args.iou_threshold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "threshold_metrics.csv", threshold_rows)
    save_csv(output_dir / "lesion_report.csv", lesion_rows)

    reason_counts = {}
    for row in lesion_rows:
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "val_list": str(Path(args.val_list).resolve()),
        "device": str(device),
        "input_size": args.input_size,
        "iou_threshold": args.iou_threshold,
        "operating_threshold": args.operating_threshold,
        "num_images": len(records),
        "num_ground_truth": num_ground_truth,
        "ap50": ap50,
        "reason_counts": reason_counts,
        "threshold_metrics": threshold_rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print()
    print(f"Images: {len(records)}")
    print(f"Ground-truth lesions: {num_ground_truth}")
    print(f"AP@{args.iou_threshold:.2f}: {ap50:.4f}")
    print(
        f"Operating threshold: {args.operating_threshold:.2f}, "
        f"reason counts: {reason_counts}"
    )
    print(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
