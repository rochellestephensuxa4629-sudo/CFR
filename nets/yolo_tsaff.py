"""YOLOX model entry point using the diagram-aligned TSAFF module."""

import torch
import torch.nn as nn

from nets.yolo import YOLOPAFPN2, YOLOXHead
from nets.tsaff_modified import TSAFF


class ConcatFusion(nn.Module):
    """Direct concat fusion with channel projection."""

    def __init__(self, channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs):
        return self.project(torch.cat(inputs, dim=1))


class YOLOPAFPN2TSAFF(YOLOPAFPN2):
    """Dual-backbone YOLOX neck with TSAFF fusion at P3, P4, and P5."""

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=(256, 512, 1024),
        depthwise=False,
        act="silu",
        tsaff_loops=2,
        qv_mode="qvgsim",
        fusion_type="tsaff",
        use_fuglrm=True,
    ):
        super().__init__(
            depth=depth,
            width=width,
            in_features=in_features,
            in_channels=in_channels,
            depthwise=depthwise,
            act=act,
            with_fusion=True,
        )

        self.fusion_type = fusion_type
        if fusion_type in ("single_ct", "single_pet"):
            self.backbone2 = None
        channels = [int(channel * width) for channel in in_channels]
        anchors = [(40, 40), (20, 20), (10, 10)]
        fusion_layers = []
        for channel, (height, width_value) in zip(channels, anchors):
            if fusion_type == "tsaff":
                layer = TSAFF(
                    channel,
                    vertical_anchors=height,
                    horizontal_anchors=width_value,
                    loops=tsaff_loops,
                    qv_mode=qv_mode,
                    use_fuglrm=use_fuglrm,
                )
            elif fusion_type == "concat":
                layer = ConcatFusion(channel)
            elif fusion_type in ("add", "single_ct", "single_pet"):
                layer = nn.Identity()
            else:
                raise ValueError(f"Unsupported fusion_type: {fusion_type}")
            fusion_layers.append(layer)

        self.fusion1, self.fusion2, self.fusion3 = fusion_layers

    def forward(self, input1, input2):
        if self.fusion_type in ("single_ct", "single_pet"):
            single_input = (
                input1 if self.fusion_type == "single_ct" else input2
            )
            out_features = self.backbone.forward(single_input)
            feature_pairs = [
                (out_features[name], None)
                for name in self.in_features
            ]
        else:
            out_features1 = self.backbone.forward(input1)
            out_features2 = self.backbone2.forward(input2)
            feature_pairs = [
                (out_features1[name], out_features2[name])
                for name in self.in_features
            ]

        fused_features = []
        fusion_layers = (self.fusion1, self.fusion2, self.fusion3)
        for layer, (feature1, feature2) in zip(
            fusion_layers,
            feature_pairs,
        ):
            if self.fusion_type in ("single_ct", "single_pet"):
                fused = feature1
            elif self.fusion_type == "add":
                fused = feature1 + feature2
            else:
                fused = layer([feature1, feature2])
            fused_features.append(fused)

        feat1, feat2, feat3 = fused_features

        P5 = self.lateral_conv0(feat3)
        P5_upsample = self.upsample(P5)
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P5_upsample = self.C3_p4(P5_upsample)

        P4 = self.reduce_conv1(P5_upsample)
        P4_upsample = self.upsample(P4)
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        P3_out = self.C3_p3(P4_upsample)

        P3_downsample = self.bu_conv2(P3_out)
        P3_downsample = torch.cat([P3_downsample, P4], 1)
        P4_out = self.C3_n3(P3_downsample)

        P4_downsample = self.bu_conv1(P4_out)
        P4_downsample = torch.cat([P4_downsample, P5], 1)
        P5_out = self.C3_n4(P4_downsample)
        return P3_out, P4_out, P5_out


class YoloBody(nn.Module):
    """Backward-compatible YOLOX body with the revised TSAFF implementation."""

    def __init__(
        self,
        num_classes,
        phi,
        tsaff_loops=2,
        qv_mode="qvgsim",
        fusion_type="tsaff",
        use_fuglrm=True,
    ):
        super().__init__()
        depth_dict = {
            "nano": 0.33,
            "tiny": 0.33,
            "s": 0.33,
            "m": 0.67,
            "l": 1.00,
            "x": 1.33,
        }
        width_dict = {
            "nano": 0.25,
            "tiny": 0.375,
            "s": 0.50,
            "m": 0.75,
            "l": 1.00,
            "x": 1.25,
        }
        depth = depth_dict[phi]
        width = width_dict[phi]
        depthwise = phi == "nano"

        self.backbone = YOLOPAFPN2TSAFF(
            depth,
            width,
            depthwise=depthwise,
            tsaff_loops=tsaff_loops,
            qv_mode=qv_mode,
            fusion_type=fusion_type,
            use_fuglrm=use_fuglrm,
        )
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, ct_image, pet_image):
        fpn_outputs = self.backbone.forward(ct_image, pet_image)
        return self.head.forward(fpn_outputs)
