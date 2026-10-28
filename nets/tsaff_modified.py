"""TSAFF implementation aligned with the architecture diagrams.

This module is intentionally self-contained so the original network files can
remain unchanged while the new TSAFF architecture is evaluated side by side.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def map_to_tokens(feature):
    return feature.flatten(2).transpose(1, 2)


def tokens_to_map(tokens, height, width):
    batch, _, channels = tokens.shape
    return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class SFM(nn.Module):
    """Spatial Feature Compression module."""

    def __init__(self, channels, output_height, output_width):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((output_height, output_width))
        self.max_pool = nn.AdaptiveMaxPool2d((output_height, output_width))
        self.pool_weights = nn.Parameter(torch.ones(2))

    def forward(self, feature):
        feature = self.project(feature)
        weights = torch.softmax(self.pool_weights, dim=0)
        return weights[0] * self.avg_pool(feature) + weights[1] * self.max_pool(feature)


class QVGSIM(nn.Module):
    """Query-Value gated semantic interaction module."""

    def __init__(
        self,
        channels,
        num_heads=8,
        mlp_ratio=4.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_target = nn.LayerNorm(channels)
        self.norm_context = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)

        self.value_gate = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, channels),
            nn.Sigmoid(),
        )
        self.query_gate = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, channels),
            nn.Sigmoid(),
        )

        hidden_channels = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(projection_dropout),
        )
        self.norm_output = nn.LayerNorm(channels)
        self.attention_dropout = nn.Dropout(attention_dropout)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.delta = nn.Parameter(torch.ones(1))

    def forward(self, target, context):
        batch, token_count, channels = target.shape

        target_norm = self.norm_target(target)
        context_norm = self.norm_context(context)

        value = self.value(target_norm)
        key = self.key(target_norm)
        query = self.query(context_norm)
        value_gate = self.value_gate(value)
        query_gate = self.query_gate(query)

        query = query.reshape(
            batch, token_count, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key = key.reshape(
            batch, token_count, self.num_heads, self.head_dim
        ).permute(0, 2, 3, 1)
        value_heads = value.reshape(
            batch, token_count, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        attention = torch.matmul(query, key) * self.scale
        attention = torch.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)

        correlated = torch.matmul(attention, value_heads)
        correlated = correlated.permute(0, 2, 1, 3).reshape(
            batch, token_count, channels
        )
        correlated = self.output(correlated)

        value_term = self.alpha * value_gate * value
        query_term = self.beta * correlated * query_gate
        fused = value_term + query_term
        refined = target + self.delta * self.ffn(self.norm_output(fused))
        return refined


class StandardCrossAttention(nn.Module):
    """Standard cross-attention without the Q-V gating branches."""

    def __init__(
        self,
        channels,
        num_heads=8,
        mlp_ratio=4.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_target = nn.LayerNorm(channels)
        self.norm_context = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)
        self.attention_dropout = nn.Dropout(attention_dropout)

        hidden_channels = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(projection_dropout),
        )
        self.norm_output = nn.LayerNorm(channels)
        self.delta = nn.Parameter(torch.ones(1))

    def forward(self, target, context):
        batch, token_count, channels = target.shape
        target_norm = self.norm_target(target)
        context_norm = self.norm_context(context)

        value = self.value(target_norm)
        key = self.key(target_norm)
        query = self.query(context_norm)

        query = query.reshape(
            batch, token_count, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key = key.reshape(
            batch, token_count, self.num_heads, self.head_dim
        ).permute(0, 2, 3, 1)
        value_heads = value.reshape(
            batch, token_count, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        attention = torch.matmul(query, key) * self.scale
        attention = torch.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)

        correlated = torch.matmul(attention, value_heads)
        correlated = correlated.permute(0, 2, 1, 3).reshape(
            batch, token_count, channels
        )
        correlated = self.output(correlated)
        return target + self.delta * self.ffn(
            self.norm_output(correlated)
        )


class IdentityRefiner(nn.Module):
    """Compatibility wrapper for the no-QVGSIM ablation."""

    def forward(self, target, context):
        return target


class IFRM(nn.Module):
    """Iterative Feature Refinement module with shared QVGSIM parameters."""

    def __init__(
        self,
        channels,
        num_heads=8,
        loops=2,
        qv_mode="qvgsim",
        mlp_ratio=4.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ):
        super().__init__()
        if loops < 1:
            raise ValueError("loops must be at least 1")

        self.loops = loops
        self.qv_mode = qv_mode
        if qv_mode == "identity":
            self.qvgsim_ct = IdentityRefiner()
            self.qvgsim_pet = IdentityRefiner()
        else:
            if qv_mode == "qvgsim":
                refiner_class = QVGSIM
            elif qv_mode == "standard_attention":
                refiner_class = StandardCrossAttention
            else:
                raise ValueError(f"Unsupported qv_mode: {qv_mode}")
            self.qvgsim_ct = refiner_class(
                channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )
            self.qvgsim_pet = refiner_class(
                channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, ct_tokens, pet_tokens):
        for _ in range(self.loops):
            ct_refined = self.qvgsim_ct(ct_tokens, pet_tokens)
            pet_refined = self.qvgsim_pet(pet_tokens, ct_tokens)
            ct_tokens, pet_tokens = ct_refined, pet_refined
        return ct_tokens, pet_tokens


class FUSM(nn.Module):
    """Fuzzy Uncertainty Suppression module."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.mu = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_sigma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.uncertainty_conv = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, feature, alignment_residual=None):
        sigma = F.softplus(self.log_sigma) + 1e-4
        membership = torch.exp(
            -((feature - self.mu) ** 2) / (2.0 * sigma ** 2)
        )
        membership_uncertainty = 4.0 * membership * (1.0 - membership)
        membership_uncertainty = membership_uncertainty.mean(
            dim=1, keepdim=True
        )

        local_mean = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
        local_second_moment = F.avg_pool2d(
            feature * feature,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        local_variance = (
            local_second_moment - local_mean * local_mean
        ).clamp_min(0.0)
        local_variance = local_variance.mean(dim=1, keepdim=True)

        if alignment_residual is None:
            residual = torch.zeros_like(membership_uncertainty)
        else:
            residual = alignment_residual.abs().mean(dim=1, keepdim=True)

        uncertainty_input = torch.cat(
            [membership_uncertainty, local_variance, residual],
            dim=1,
        )
        uncertainty = self.uncertainty_conv(uncertainty_input)
        confidence = 1.0 - uncertainty

        high_confidence = feature * confidence
        low_confidence = feature * uncertainty
        refined = torch.cat([high_confidence, low_confidence], dim=1)
        refined = self.refine(refined)
        return feature + refined, uncertainty


class TSAFF(nn.Module):
    """Two-Stage Adaptive Feature Fusion module from the paper diagram."""

    def __init__(
        self,
        channels,
        vertical_anchors=40,
        horizontal_anchors=40,
        num_heads=8,
        loops=2,
        qv_mode="qvgsim",
        use_fuglrm=True,
        mlp_ratio=4.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ):
        super().__init__()
        self.vertical_anchors = vertical_anchors
        self.horizontal_anchors = horizontal_anchors
        self.use_fuglrm = use_fuglrm
        token_count = vertical_anchors * horizontal_anchors

        self.sfm_ct = SFM(
            channels,
            output_height=vertical_anchors,
            output_width=horizontal_anchors,
        )
        self.sfm_pet = SFM(
            channels,
            output_height=vertical_anchors,
            output_width=horizontal_anchors,
        )
        self.position_ct = nn.Parameter(torch.zeros(1, token_count, channels))
        self.position_pet = nn.Parameter(torch.zeros(1, token_count, channels))

        self.ifrm = IFRM(
            channels,
            num_heads=num_heads,
            loops=loops,
            qv_mode=qv_mode,
            mlp_ratio=mlp_ratio,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )

        self.zeta = nn.Parameter(torch.ones(1))
        self.epsilon = nn.Parameter(torch.ones(1))
        self.phi = nn.Parameter(torch.ones(1))
        self.eta = nn.Parameter(torch.ones(1))

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        if use_fuglrm:
            self.fusm_ct = FUSM(channels)
            self.fusm_pet = FUSM(channels)
            self.fusm_fused = FUSM(channels)

    def forward(self, inputs, num=None):
        ct_feature, pet_feature = inputs
        batch, channels, input_height, input_width = ct_feature.shape

        ct_compressed = self.sfm_ct(ct_feature)
        pet_compressed = self.sfm_pet(pet_feature)
        compressed_height = ct_compressed.shape[-2]
        compressed_width = ct_compressed.shape[-1]

        ct_tokens = map_to_tokens(ct_compressed) + self.position_ct
        pet_tokens = map_to_tokens(pet_compressed) + self.position_pet

        ct_refined, pet_refined = self.ifrm(ct_tokens, pet_tokens)

        ct_mixed = self.zeta * ct_tokens + self.epsilon * ct_refined
        pet_mixed = self.phi * pet_tokens + self.eta * pet_refined

        ct_map = tokens_to_map(
            ct_mixed,
            compressed_height,
            compressed_width,
        )
        pet_map = tokens_to_map(
            pet_mixed,
            compressed_height,
            compressed_width,
        )
        fused = self.fusion(torch.cat([ct_map, pet_map], dim=1))
        if self.use_fuglrm:
            ct_alignment = tokens_to_map(
                ct_refined,
                compressed_height,
                compressed_width,
            )
            pet_alignment = tokens_to_map(
                pet_refined,
                compressed_height,
                compressed_width,
            )
            ct_map, _ = self.fusm_ct(
                ct_map,
                alignment_residual=ct_map - ct_alignment,
            )
            pet_map, _ = self.fusm_pet(
                pet_map,
                alignment_residual=pet_map - pet_alignment,
            )
            fused = self.fusion(torch.cat([ct_map, pet_map], dim=1))
            fused, _ = self.fusm_fused(
                fused,
                alignment_residual=ct_map - pet_map,
            )
        return F.interpolate(
            fused,
            size=(input_height, input_width),
            mode="bilinear",
            align_corners=False,
        )
