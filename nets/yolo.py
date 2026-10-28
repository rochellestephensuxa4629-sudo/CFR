#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from nets.common import TSAFF


def generate_cosine_positional_encoding(d_model, height, width):
    pe = torch.zeros(d_model, height, width)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_y, pos_x = torch.meshgrid(torch.arange(0., height), torch.arange(0., width))
    pe[0::2, :, :] = torch.sin(pos_y * div_term[:, None, None])
    pe[1::2, :, :] = torch.cos(pos_x * div_term[:, None, None])
    return pe
    
class CrossSpectrumAttentionWithPosition(nn.Module):
    def __init__(self, in_channels, num_stages):
        super(CrossSpectrumAttentionWithPosition, self).__init__()
        self.num_stages = num_stages
        self.attention_stages_k1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])
        
        self.attention_stages_k3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=5),
                nn.Conv2d(in_channels, in_channels, kernel_size=5),
                nn.Conv2d(in_channels, in_channels, kernel_size=5),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])
        """
        self.attention_stages_k3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3),
                nn.Conv2d(in_channels, in_channels, kernel_size=3),
                nn.Conv2d(in_channels, in_channels, kernel_size=3),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])
        """
        self.attention_stages_k7 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=7),
                nn.Conv2d(in_channels, in_channels, kernel_size=7),
                nn.Conv2d(in_channels, in_channels, kernel_size=7),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_stages)
        ])
        
        self.compression_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels*3, in_channels, kernel_size=1),
            )
            for _ in range(num_stages)
        ])
        
        
        self.proj = nn.Conv2d(in_channels, 64, kernel_size=1)
        

    def forward(self, x1, x2):  
        # hack : future implementation with learned pos_encoding ? 
        pos_encoding1 = generate_cosine_positional_encoding(x1.size(1), x1.size(2), x1.size(3))
        pos_encoding2 = generate_cosine_positional_encoding(x2.size(1), x2.size(2), x2.size(3))
        
        # additive (hack ?) fused pos_encoding 
        combined_pos_encoding = pos_encoding1 + pos_encoding2
        combined_pos_encoding = combined_pos_encoding.to(x1.device)
        
        
        for i in range(self.num_stages): # loop for hierarchical self/cross attention steps 
        # hierarchical attention + learned attention fusion (convolution based procedure)
            query_conv1, key_conv1, value_conv1, softmax1 = self.attention_stages_k1[i]
            query1 = query_conv1(x1) + combined_pos_encoding
            key1 = key_conv1(x2) + combined_pos_encoding
            value1 = value_conv1(x2)
            attn_weights1 = softmax1(torch.matmul(query1, key1.transpose(-1, -2)))
            fused_features_k1 = torch.matmul(attn_weights1, value1)
            
            query_conv3, key_conv3, value_conv3, softmax3 = self.attention_stages_k3[i]
            query3 = query_conv3(x1) 
            key3 = key_conv3(x2)
            value3 = value_conv3(x2)
            attn_weights3 = softmax3(torch.matmul(query3, key3.transpose(-1, -2)))
            fused_features_k3 = torch.matmul(attn_weights3, value3)
            
            query_conv7, key_conv7, value_conv7, softmax7 = self.attention_stages_k7[i]
            query7 = query_conv7(x1) 
            key7 = key_conv7(x2) 
            value7 = value_conv7(x2)
            attn_weights7 = softmax7(torch.matmul(query7, key7.transpose(-1, -2)))
            fused_features_k7 = torch.matmul(attn_weights7, value7)

            #  target features shape 
            target_height = fused_features_k1.size(2)
            target_width = fused_features_k1.size(3)

            # reshaping the different attention maps before concatenation : interpolation (not-learned) 
            k3_resized = F.interpolate(fused_features_k3, size=(target_height, target_width), mode='bilinear', align_corners=False)
            k7_resized = F.interpolate(fused_features_k7, size=(target_height, target_width), mode='bilinear', align_corners=False)

            fused_features = torch.cat((fused_features_k1, k3_resized, k7_resized), 1) # stacking 
            fused_features = self.compression_convs[i](fused_features) # learned operation: compress to the standard features shape 
            
            # Update x1 and x2 for the next stage
            x1, x2 = fused_features, fused_features
            
        # projects to the reduced dimensionnality (legacy)
        #x1_conv = self.vis_proj(x1_value)
        #x2_conv = self.ir_proj(x2_value)
        
        # features concatenation (legacy) 
        #features = torch.cat((x1_conv, x2_conv, fused_proj), 1)
        
        fused_features = self.proj(fused_features) # projection : learned reshaping for dimensionality compression
        return fused_features


class CIA(nn.Module):
    """
    简化版跨模态交互与聚合模块（适用于两个模态）
    输入形状：(B, N, H, W)
    B: batch size
    N: 通道数（特征维度）
    H, W: 空间维度
    """
    
    def __init__(self, in_channels, reduction_ratio=2):
        super(CIA, self).__init__()
        
        self.in_channels = in_channels
        
        # 上下文注入模块 (Context Injection Module)
        # 注意力分支：生成注意力图
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 主融合分支：处理次要模态特征
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # 可选的残差连接后的进一步处理
        self.post_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, main_feat, minor_feat):
        """
        前向传播
        
        Args:
            main_feat: 主要模态特征 (B, N, H, W)
            minor_feat: 次要模态特征 (B, N, H, W)
            
        Returns:
            增强后的主要模态特征 (B, N, H, W)
        """
        # 1. 计算注意力图
        # 拼接主要模态和次要模态特征
        combined = torch.cat([main_feat, minor_feat], dim=1)  # (B, 2*N, H, W)
        attention_map = self.attention_conv(combined)  # (B, N, H, W)
        
        # 2. 处理次要模态特征
        minor_transformed = self.fusion_conv(minor_feat)  # (B, N, H, W)
        
        # 3. 使用注意力图加权次要模态特征
        weighted_minor = minor_transformed * attention_map  # (B, N, H, W)
        
        # 4. 与主要模态残差连接（保持主要模态的主导地位）
        fused_feat = weighted_minor + main_feat  # (B, N, H, W)
        
        # 5. 可选的进一步处理
        output =  self.post_fusion(fused_feat) + fused_feat  # 残差连接
        
        return output


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)  

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P3_downsample   = self.bu_conv2(P3_out) 
        #-------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P3_downsample   = torch.cat([P3_downsample, P4], 1) 
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P4_out          = self.C3_n3(P3_downsample) 

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P4_downsample   = self.bu_conv1(P4_out)
        #-------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P4_downsample   = torch.cat([P4_downsample, P5], 1)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P5_out          = self.C3_n4(P4_downsample)

        return (P3_out, P4_out, P5_out)


class YOLOPAFPN2(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu", with_fusion=True):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone     = CSPDarknet(depth, width, depthwise = depthwise, act = act, with_fusion=True)
        self.backbone2       = CSPDarknet(depth, width, depthwise = depthwise, act = act, with_fusion=True)
        self.fusion1 = TSAFF(256, 40, 40, n_layer=1)
        self.fusion2 = TSAFF(512, 20, 20, n_layer=1)
        self.fusion3 = TSAFF(1024, 10, 10,  n_layer=1)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
        # self.fusion1 = CrossSpectrumAttentionWithPosition(in_channels[0], 2)
        # self.fusion2 = CrossSpectrumAttentionWithPosition(in_channels[1], 2)
        # self.fusion3 = CrossSpectrumAttentionWithPosition(in_channels[2], 2)
        # redim_corr_list=[]
        # for i in range(len(in_channels)): 
        #     redim_corr_list.append(nn.Sequential(
        #             nn.Conv2d(2*in_channels[i]+64, in_channels[i], kernel_size=1),
        #             nn.GroupNorm(32, in_channels[i])))
        # self.redim_corr = nn.ModuleList(redim_corr_list)
        # self.CIA1  = CIA(in_channels[0])
        # self.CIA2  = CIA(in_channels[1])
        # self.CIA3  = CIA(in_channels[2])
        
    def forward(self, input1, input2):
        out_features1            = self.backbone.forward(input1)
        out_features2            = self.backbone2.forward(input2)

        [feat1, feat2, feat3]   = [out_features1[f] for f in self.in_features]
        [feat11, feat22, feat33]   = [out_features2[f] for f in self.in_features]
        feat1 = self.fusion1([feat1, feat11],1)
        feat2 = self.fusion2([feat2, feat22],2)
        feat3 = self.fusion3([feat3, feat33],3)

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)  

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P3_downsample   = self.bu_conv2(P3_out) 
        #-------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P3_downsample   = torch.cat([P3_downsample, P4], 1) 
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P4_out          = self.C3_n3(P3_downsample) 

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P4_downsample   = self.bu_conv1(P4_out)
        #-------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P4_downsample   = torch.cat([P4_downsample, P5], 1)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P5_out          = self.C3_n4(P4_downsample)

        return (P3_out, P4_out, P5_out)



class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 

        self.backbone   = YOLOPAFPN2(depth, width, depthwise=depthwise)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x,y):
        fpn_outs    = self.backbone.forward(x,y)
        outputs     = self.head.forward(fpn_outs)
        return outputs
