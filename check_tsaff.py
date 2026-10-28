#!/usr/bin/env python3
"""Smoke-test the revised TSAFF model with CT and PET tensors."""

import argparse

import torch

from nets.yolo_tsaff import YoloBody


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi", default="l")
    parser.add_argument("--classes", type=int, default=1)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument(
        "--qv-mode",
        choices=("qvgsim", "standard_attention", "identity"),
        default="qvgsim",
    )
    parser.add_argument(
        "--fusion-type",
        choices=("tsaff", "add", "concat", "single_ct", "single_pet"),
        default="tsaff",
    )
    parser.add_argument("--no-fuglrm", action="store_true")
    parser.add_argument("--size", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloBody(
        args.classes,
        args.phi,
        tsaff_loops=args.loops,
        qv_mode=args.qv_mode,
        fusion_type=args.fusion_type,
        use_fuglrm=not args.no_fuglrm,
    ).to(device)
    model.eval()

    ct_image = torch.randn(1, 3, args.size, args.size, device=device)
    pet_image = torch.randn(1, 3, args.size, args.size, device=device)
    with torch.no_grad():
        outputs = model(ct_image, pet_image)

    for index, output in enumerate(outputs, start=1):
        print(f"P{index + 2}: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
