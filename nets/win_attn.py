# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "SwinUNETR",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]

class HeadSpecificQueryDependentGating(nn.Module):
    """
    真正的头特定+查询依赖门控：
    每个头使用自己的查询（query）计算门控分数
    """
    
    def __init__(self, num_heads, head_dim, gate_type='elementwise'):
        """
        Args:
            num_heads: 注意力头数量
            head_dim: 每个头的维度
            gate_type: 'elementwise'或'headwise'
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gate_type = gate_type
        
        # 为每个头创建独立的门控参数
        # 参数形状: [num_heads, head_dim, gate_output_dim]
        if gate_type == 'elementwise':
            gate_output_dim = head_dim  # 逐元素门控：每个维度独立
        elif gate_type == 'headwise':
            gate_output_dim = 1  # 逐头门控：整个头一个标量
        
        self.gate_weights = nn.Parameter(
            torch.zeros(num_heads, head_dim, gate_output_dim)
        )
        nn.init.normal_(self.gate_weights, std=0.02)
        
    def forward(self, attention_output, query):
        """
        Args:
            attention_output: 注意力输出 [B, num_heads, N, head_dim]
            query: 查询向量 [B, num_heads, N, head_dim]
        Returns:
            gated_output: 门控调制后的输出
        """
        B, num_heads, N, head_dim = attention_output.shape
        
        # 使用每个头的查询计算门控分数
        # query: [B, num_heads, N, head_dim]
        # gate_weights: [num_heads, head_dim, gate_output_dim]
        # gate_scores: [B, num_heads, N, gate_output_dim]
        gate_scores = torch.einsum('bhnc,hcd->bhnd', query, self.gate_weights)
        gate_scores = torch.sigmoid(gate_scores)
        
        # 对于逐头门控，需要广播到所有维度
        if self.gate_type == 'headwise':
            gate_scores = gate_scores.unsqueeze(-1)  # [B, num_heads, N, 1]
            gate_scores = gate_scores.expand(-1, -1, -1, head_dim)  # [B, num_heads, N, head_dim]
        
        # 应用乘性门控
        gated_output = attention_output * gate_scores
        
        return gated_output


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()  # length 4 or 5 only
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    else:  # if len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)

    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


# class WindowAttention(nn.Module):
#     """
#     Window based multi-head self attention module with relative position bias based on: "Liu et al.,
#     Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
#     <https://arxiv.org/abs/2103.14030>"
#     https://github.com/microsoft/Swin-Transformer
#     """

#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         window_size: Sequence[int],
#         qkv_bias: bool = False,
#         attn_drop: float = 0.0,
#         proj_drop: float = 0.0,
#     ) -> None:
#         """
#         Args:
#             dim: number of feature channels.
#             num_heads: number of attention heads.
#             window_size: local window size.
#             qkv_bias: add a learnable bias to query, key, value.
#             attn_drop: attention dropout rate.
#             proj_drop: dropout rate of output.
#         """

#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim**-0.5
#         mesh_args = torch.meshgrid.__kwdefaults__

#         if len(self.window_size) == 3:
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros(
#                     (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
#                     num_heads,
#                 )
#             )
#             coords_d = torch.arange(self.window_size[0])
#             coords_h = torch.arange(self.window_size[1])
#             coords_w = torch.arange(self.window_size[2])
#             if mesh_args is not None:
#                 coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
#             else:
#                 coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
#             coords_flatten = torch.flatten(coords, 1)
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#             relative_coords[:, :, 0] += self.window_size[0] - 1
#             relative_coords[:, :, 1] += self.window_size[1] - 1
#             relative_coords[:, :, 2] += self.window_size[2] - 1
#             relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
#             relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
#         elif len(self.window_size) == 2:
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
#             )
#             coords_h = torch.arange(self.window_size[0])
#             coords_w = torch.arange(self.window_size[1])
#             if mesh_args is not None:
#                 coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
#             else:
#                 coords = torch.stack(torch.meshgrid(coords_h, coords_w))
#             coords_flatten = torch.flatten(coords, 1)
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#             relative_coords[:, :, 0] += self.window_size[0] - 1
#             relative_coords[:, :, 1] += self.window_size[1] - 1
#             relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

#         relative_position_index = relative_coords.sum(-1)
#         self.register_buffer("relative_position_index", relative_position_index)
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         trunc_normal_(self.relative_position_bias_table, std=0.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask):
#         b, n, c = x.shape
#         qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         q = q * self.scale
#         attn = q @ k.transpose(-2, -1)
#         relative_position_bias = self.relative_position_bias_table[
#             self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore[operator]
#         ].reshape(n, n, -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#         attn = attn + relative_position_bias.unsqueeze(0)
#         if mask is not None:
#             nw = mask.shape[0]
#             attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, n, n)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)

#         attn = self.attn_drop(attn).to(v.dtype)
#         x = (attn @ v).transpose(1, 2).reshape(b, n, c)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


class WindowAttention(nn.Module):
    """Window attention with cross-attention support"""
    
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        use_gating=True,
        gate_type='elementwise',
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.gate_type = gate_type
        self.use_gating = use_gating
        
        # 相对位置编码
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        self.register_buffer("relative_position_index", self.get_relative_position_index(window_size))
        
        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        # ========== 头特定查询依赖门控模块 ==========
        if use_gating:
            self.gating_module = HeadSpecificQueryDependentGating(
                num_heads=num_heads,
                head_dim=head_dim,
                gate_type=gate_type
            )
        # ===========================================

        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def get_relative_position_index(self, window_size):
        """生成相对位置索引"""
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        return relative_coords.sum(-1)

    def forward(self, x, mask=None, context=None):
        """
        Forward pass
        Args:
            x: input features (query)
            mask: attention mask
            context: context features for cross-attention (if provided)
        """
        B_, N, C = x.shape
        
        if context is not None:
            # 交叉注意力：x作为query，context作为key和value
            q = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q = q[0]  # query from x
            
            kv = self.qkv(context).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[1], kv[2]  # key, value from context
        else:
            # 自注意力
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        
        # 计算注意力输出
        attn_output = attn @ v  # [B, num_heads, N, head_dim]
        
        # ========== 应用头特定查询依赖门控 ==========
        if self.use_gating:
            # 关键：使用每个头的查询（q）作为门控输入
            # attn_output: [B, num_heads, N, head_dim]
            # q: [B, num_heads, N, head_dim]
            attn_output = self.gating_module(attn_output, q)
        # =========================================
        
        # 转置并reshape
        x = attn_output.transpose(1, 2).reshape(B_, N, C)
        
        # 最终投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block with cross-attention support for multi-modal inputs.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
        cross_attention: bool = True,  # 新增：是否使用交叉注意力
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            cross_attention: whether to use cross-attention for multi-modal inputs.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.cross_attention = cross_attention
        
        # 自注意力路径的归一化层
        self.norm1 = norm_layer(dim)
        
        # 如果是交叉注意力，需要为第二种模态也创建归一化层
        if cross_attention:
            self.norm1_context = norm_layer(dim)
        
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_gating=True,
            gate_type='elementwise',
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix, context=None):
        """
        Forward part 1: attention mechanism
        Args:
            x: input tensor of shape (B, H, W, C)
            mask_matrix: attention mask
            context: context tensor for cross-attention, shape (B, H, W, C)
        """
        x_shape = x.size()
        # print(x.size(), context.size())
        if self.cross_attention and context is not None:
            # 交叉注意力模式
            x_norm = self.norm1(x)
            context_norm = self.norm1_context(context)
            
            b, h, w, c = x_norm.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            
            # 对query和context进行相同的padding
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            
            x_padded = F.pad(x_norm, (0, 0, pad_l, pad_r, pad_t, pad_b))
            context_padded = F.pad(context_norm, (0, 0, pad_l, pad_r, pad_t, pad_b))
            
            _, hp, wp, _ = x_padded.shape
            dims = [b, hp, wp]

            # 应用相同的shift操作
            if any(i > 0 for i in shift_size):
                if len(x_shape) == 5:
                    shifted_x = torch.roll(x_padded, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
                    shifted_context = torch.roll(context_padded, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
                elif len(x_shape) == 4:
                    shifted_x = torch.roll(x_padded, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
                    shifted_context = torch.roll(context_padded, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
                attn_mask = mask_matrix
            else:
                shifted_x = x_padded
                shifted_context = context_padded
                attn_mask = None
            
            # 划分窗口
            x_windows = window_partition(shifted_x, window_size)  # query
            context_windows = window_partition(shifted_context, window_size)  # key, value
            
            # 交叉注意力：x作为query，context作为key和value
            attn_windows = self.attn(x_windows, mask=attn_mask, context=context_windows)
            
        else:
            # 原始的自注意力模式
            x_norm = self.norm1(x)
            
            b, h, w, c = x_norm.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x_padded = F.pad(x_norm, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x_padded.shape
            dims = [b, hp, wp]

            if any(i > 0 for i in shift_size):
                if len(x_shape) == 5:
                    shifted_x = torch.roll(x_padded, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
                elif len(x_shape) == 4:
                    shifted_x = torch.roll(x_padded, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
                attn_mask = mask_matrix
            else:
                shifted_x = x_padded
                attn_mask = None
            
            x_windows = window_partition(shifted_x, window_size)
            attn_windows = self.attn(x_windows, mask=attn_mask)
        
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_b > 0 or pad_r > 0:
                x = x[:, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        """Forward part 2: MLP"""
        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        """Load pretrained weights"""
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight", "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight", "attn.qkv.bias",
            "attn.proj.weight", "attn.proj.bias",
            "norm2.weight", "norm2.bias",
            "mlp.fc1.weight", "mlp.fc1.bias",
            "mlp.fc2.weight", "mlp.fc2.bias",
        ]
        
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            
            # 如果是交叉注意力，为context归一化层复制相同的权重
            if self.cross_attention:
                self.norm1_context.weight.copy_(weights["state_dict"][root + block_names[0]])
                self.norm1_context.bias.copy_(weights["state_dict"][root + block_names[1]])
            
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def forward(self, x, mask_matrix, context=None):
        """
        Forward pass with support for cross-attention
        Args:
            x: input tensor (query modality)
            mask_matrix: attention mask
            context: context tensor for cross-attention (key/value modality)
        """
        shortcut = x
        
        if self.use_checkpoint:
            if self.cross_attention:
                x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, context, use_reentrant=False)
            else:
                x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix, context)
        
        x = shortcut + self.drop_path(x)
        
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        
        return x

# MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
        cross_attention = True
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size= self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    cross_attention = cross_attention
                )
 
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

    def forward(self, x, y = None):
        x_shape = x.size()
        # print(x_shape)
        b, c, h, w = x_shape
        window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
        # print(x.size(), y.size())
        x = rearrange(x, "b c h w -> b h w c")
        if y is not None:
            y = rearrange(y, "b c h w -> b h w c")
        # print(x.size(), y.size())
        hp = int(np.ceil(h / window_size[0])) * window_size[0]
        wp = int(np.ceil(w / window_size[1])) * window_size[1]
        attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
        x = self.blocks(x, attn_mask, y)
        x = x.view(b, h, w, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b h w c -> b c h w")
        
        return x

