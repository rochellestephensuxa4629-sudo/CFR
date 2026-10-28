import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.win_attn import BasicLayer
import math

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
        
        
        # self.proj = nn.Conv2d(in_channels, 64, kernel_size=1)
        

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
        
        
        # fused_features = self.proj(fused_features) # projection : learned reshaping for dimensionality compression
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



class LGU(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size = 1),
                              nn.ReLU(),
                              nn.Conv2d(input_channel, input_channel, kernel_size = 1),
                              nn.Sigmoid())
        
    def forward(self, x):
        return x * self.layer(x)

        
class CrossAttention(nn.Module):
    """
    使用标准神经网络层实现的门控交叉注意力
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 use_gating=True, gate_type='elementwise'):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_gating = use_gating
        self.gate_type = gate_type

        # Linear projections for query, key, value
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # ========== 共享门控网络 ==========
        if use_gating:
            if gate_type == 'elementwise':
                # 逐元素门控网络
                self.gate_net = nn.Sequential(
                    nn.Linear(self.head_dim, self.head_dim * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.head_dim * 4, self.head_dim),
                    nn.Sigmoid()
                )
            elif gate_type == 'headwise':
                # 逐头门控网络
                self.gate_net = nn.Sequential(
                    nn.Linear(self.head_dim, self.head_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.head_dim // 2, 1),
                    nn.Sigmoid()
                )
        # ================================

    def forward(self, query, key_value):
        """
        Cross attention with standard gating
        Args:
            query: [B, N, C] query from modality A
            key_value: [B, M, C] key and value from modality B
        """
        B, N, C = query.shape
        M = key_value.shape[1]
        H = self.num_heads
        D = self.head_dim
        
        # Project to Q, K, V
        q = self.wq(query).reshape(B, N, H, D).permute(0, 2, 1, 3)  # [B, H, N, D]
        k = self.wk(key_value).reshape(B, M, H, D).permute(0, 2, 1, 3)  # [B, H, M, D]
        v = self.wv(key_value).reshape(B, M, H, D).permute(0, 2, 1, 3)  # [B, H, M, D]
        
        # Attention mechanism
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, M]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        attn_output = attn @ v  # [B, H, N, D]
        
        # ========== 应用门控机制 ==========
        if self.use_gating:
            # 展平查询以便批量处理
            q_flat = q.reshape(-1, D)  # [B*H*N, D]
            
            # 计算门控分数
            gate_scores_flat = self.gate_net(q_flat)  # [B*H*N, output_dim]
            
            if self.gate_type == 'elementwise':
                # 恢复形状并应用门控
                gate_scores = gate_scores_flat.reshape(B, H, N, D)  # [B, H, N, D]
                attn_output = attn_output * gate_scores
                
            elif self.gate_type == 'headwise':
                # gate_scores_flat: [B*H*N, 1]
                # 恢复形状: [B, H, N, 1]
                gate_scores = gate_scores_flat.reshape(B, H, N, 1)
                
                # 对每个头取所有位置的平均值
                gate_scores_mean = gate_scores.mean(dim=2, keepdim=True)  # [B, H, 1, 1]
                
                # 广播到所有位置和维度
                gate_scores_broadcast = gate_scores_mean.expand(-1, -1, N, D)  # [B, H, N, D]
                
                # 应用门控
                attn_output = attn_output * gate_scores_broadcast
        # =================================
        
        # 重新排列维度并应用输出投影
        x = attn_output.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SelfAttention(nn.Module):
    """
    使用标准神经网络层实现的门控自注意力
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 use_gating=True, gate_type='elementwise', reduction_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_gating = use_gating
        self.gate_type = gate_type

        # Combined QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # ========== 使用标准神经网络层实现门控 ==========
        if use_gating:
            if gate_type == 'elementwise':
                # 逐元素门控：每个头输出head_dim维门控分数
                gate_output_dim = self.head_dim
            elif gate_type == 'headwise':
                # 逐头门控：每个头输出1个标量门控值
                gate_output_dim = 1
            
            # 为每个头创建独立的门控网络
            # 使用ModuleList存储每个头的门控层
            self.gate_nets = nn.ModuleList()
            
            for _ in range(num_heads):
                # 每个头的门控网络
                if gate_type == 'elementwise':
                    # 逐元素门控：输入head_dim，输出head_dim
                    gate_net = nn.Sequential(
                        nn.Linear(self.head_dim, self.head_dim // reduction_ratio),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.head_dim // reduction_ratio, self.head_dim),
                        nn.Sigmoid()
                    )
                elif gate_type == 'headwise':
                    # 逐头门控：输入head_dim，输出1
                    gate_net = nn.Sequential(
                        nn.Linear(self.head_dim, self.head_dim // reduction_ratio),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.head_dim // reduction_ratio, 1),
                        nn.Sigmoid()
                    )
                
                self.gate_nets.append(gate_net)
        # ================================================

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个都是 [B, H, N, D]
        
        # 注意力机制
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力到values上
        attn_output = attn @ v  # [B, H, N, D]
        
        # ========== 应用门控机制 ==========
        if self.use_gating:
            # 对每个头应用门控
            gated_heads = []
            
            for h in range(H):
                # 获取当前头的查询和输出
                head_query = q[:, h]  # [B, N, D]
                head_output = attn_output[:, h]  # [B, N, D]
                
                # 使用当前头的查询计算门控分数
                # 将每个位置独立计算门控分数
                # 我们可以使用卷积或全连接，这里选择全连接
                
                if self.gate_type == 'elementwise':
                    # 逐位置计算门控分数
                    # head_query: [B, N, D]
                    # 我们希望在每个位置上独立计算门控分数
                    B_head, N_head, D_head = head_query.shape
                    
                    # 展平为 [B*N, D] 以便批量处理
                    head_query_flat = head_query.reshape(-1, D_head)
                    
                    # 通过门控网络
                    gate_scores_flat = self.gate_nets[h](head_query_flat)  # [B*N, D]
                    
                    # 恢复形状
                    gate_scores = gate_scores_flat.reshape(B_head, N_head, D_head)
                    
                    # 应用门控
                    gated_head = head_output * gate_scores
                    
                elif self.gate_type == 'headwise':
                    # 逐头门控：每个位置使用相同的门控值
                    # head_query: [B, N, D]
                    # 计算每个位置的门控分数，然后取平均作为该头的门控分数
                    
                    # 展平并计算门控分数
                    head_query_flat = head_query.reshape(-1, D)
                    gate_scores_flat = self.gate_nets[h](head_query_flat)  # [B*N, 1]
                    
                    # 恢复形状并取所有位置的平均值
                    gate_scores_reshaped = gate_scores_flat.reshape(B, N, 1)
                    gate_score = gate_scores_reshaped.mean(dim=1, keepdim=True)  # [B, 1, 1]
                    
                    # 广播到所有位置和维度
                    gate_scores = gate_score.expand(-1, N, D)  # [B, N, D]
                    
                    # 应用门控
                    gated_head = head_output * gate_scores
                
                gated_heads.append(gated_head)
            
            # 重新组合所有头
            attn_output = torch.stack(gated_heads, dim=1)  # [B, H, N, D]
        # =================================
        
        # 重新排列维度并应用输出投影
        x = attn_output.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class SpatialToToken(nn.Module):
    """Convert spatial features (B, C, H, W) to tokens (B, N, C)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, N, C]
        return x


class TokenToSpatial(nn.Module):
    """Convert tokens (B, N, C) back to spatial features (B, C, H, W)"""
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W
    
    def forward(self, x):
        # x: [B, N, C] -> [B, C, H, W]
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, self.H, self.W)
        return x


class AMF(nn.Module):
    """
    Attention-based Multi-modal Fusion module for image inputs
    Based on the paper: MUVF-YOLOX: A Multi-modal Ultrasound Video Fusion Network for Renal Tumor Diagnosis
    """
    def __init__(self, dim, H, W, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.H = H
        self.W = W
        self.num_heads = num_heads
        
        # Convert between spatial and token representations
        self.spatial_to_token = SpatialToToken()
        self.token_to_spatial = TokenToSpatial(H, W)


        # self.CIA1  = CIA(dim)
        # self.CIA2  = CIA(dim)
        # Cross attention modules for both directions
        # self.cross_attn_b_to_c = CrossAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop, use_gating=False, gate_type='elementwise')
        # self.cross_attn_c_to_b = CrossAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop, use_gating=False, gate_type='elementwise')
        self.cross_attn_b_to_c = CrossSpectrumAttentionWithPosition(dim, 1)
        self.cross_attn_c_to_b = CrossSpectrumAttentionWithPosition(dim, 1)
        # self.cross_attn_b_to_c = BasicLayer(dim = dim,
        #                                     depth = None,
        #                                     num_heads = num_heads,
        #                                     window_size = [8,8],
        #                                     drop_path = 0,
        #                                     mlp_ratio = 4.0,
        #                                     qkv_bias = False,
        #                                     drop = 0.0,
        #                                     attn_drop = 0.0,
        #                                     norm_layer = nn.LayerNorm,
        #                                     downsample = None,
        #                                     use_checkpoint = False,
        #                                     cross_attention  =True)
        # self.cross_attn_c_to_b = BasicLayer(dim = dim,
        #                                     depth = None,
        #                                     num_heads = num_heads,
        #                                     window_size = [8,8],
        #                                     drop_path = 0,
        #                                     mlp_ratio = 4.0,
        #                                     qkv_bias = False,
        #                                     drop = 0.0,
        #                                     attn_drop = 0.0,
        #                                     norm_layer = nn.LayerNorm,
        #                                     downsample = None,
        #                                     use_checkpoint = False,
        #                                     cross_attention  =True)
        # Self attention modules for both modalities
        self.self_attn_b = SelfAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop, use_gating=False, gate_type='elementwise')
        self.self_attn_c = SelfAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop, use_gating=False, gate_type='elementwise')
        # self.self_attn_b = BasicLayer(dim = dim,
        #                                     depth = None,
        #                                     num_heads = num_heads,
        #                                     window_size = [8,8],
        #                                     drop_path = 0,
        #                                     mlp_ratio = 4.0,
        #                                     qkv_bias = False,
        #                                     drop = 0.0,
        #                                     attn_drop = 0.0,
        #                                     norm_layer = nn.LayerNorm,
        #                                     downsample = None,
        #                                     use_checkpoint = False,
        #                                     cross_attention  = False)
        # self.self_attn_c = BasicLayer(dim = dim,
        #                                     depth = None,
        #                                     num_heads = num_heads,
        #                                     window_size = [8,8],
        #                                     drop_path = 0,
        #                                     mlp_ratio = 4.0,
        #                                     qkv_bias = False,
        #                                     drop = 0.0,
        #                                     attn_drop = 0.0,
        #                                     norm_layer = nn.LayerNorm,
        #                                     downsample = None,
        #                                     use_checkpoint = False,
        #                                     cross_attention  = False)
        
        # Layer normalization
        self.norm1_b = nn.LayerNorm(dim)
        self.norm1_c = nn.LayerNorm(dim)
        self.norm2_b = nn.LayerNorm(dim)
        self.norm2_c = nn.LayerNorm(dim)
        
        # MLP for final feature fusion
        mlp_hidden_dim = int(dim * 4)
        self.mlp = nn.Sequential(
            # nn.Linear(dim * 3, dim),
            nn.Linear(dim * 3, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(proj_drop)
        )
        
        self.norm3 = nn.LayerNorm(dim)
        self.lgu1 = LGU(dim)
        self.lgu2 = LGU(dim)
        self.lgu3 = LGU(dim)
        
    def forward(self, f_b, f_c):
        """
        Forward pass of AMF module for image inputs
        Args:
            f_b: B-mode features [B, C, H, W]
            f_c: CEUS-mode features [B, C, H, W]
        Returns:
            fused_features: [B, C, H, W]
        """
        B, C, H, W = f_b.shape
        
        # Convert spatial features to tokens
        # f_b_tokens = self.spatial_to_token(f_b)  # [B, H*W, C]
        # f_c_tokens = self.spatial_to_token(f_c)  # [B, H*W, C]
        # f_b_tokens = f_b  # [B, H*W, C]
        # f_c_tokens = f_c  # [B, H*W, C]
        
        # Extract modality-invariant features using cross-attention
        # Eq(1): F_invar = Softmax(Q_B K_C^T / sqrt(d)) V_C + Softmax(Q_C K_B^T / sqrt(d)) V_B
        # print(B, C, H, W )
        f_invar_b2c = self.cross_attn_b_to_c(f_b, f_c)
        f_invar_c2b = self.cross_attn_c_to_b(f_c, f_b)
        # f_invar_b2c = self.cross_attn_b_to_c(f_b_tokens, f_c_tokens)
        # f_invar_c2b = self.cross_attn_c_to_b(f_c_tokens, f_b_tokens)
        f_invar = f_invar_b2c + f_invar_c2b  # modality-invariant features


        
        # Extract modality-specific features using self-attention with residual connection
        # Eq(2): F_B_spec = Softmax(Q_B K_B^T / sqrt(d)) V_B + F_B
        # Eq(3): F_C_spec = Softmax(Q_C K_C^T / sqrt(d)) V_C + F_C
        f_b_tokens = self.spatial_to_token(f_b)  # [B, H*W, C]
        f_c_tokens = self.spatial_to_token(f_c)  # [B, H*W, C]
        f_b_spec = self.self_attn_b(self.norm2_b(f_b_tokens)) + f_b_tokens
        f_c_spec = self.self_attn_c(self.norm2_c(f_c_tokens)) + f_c_tokens
        # f_b_spec = self.self_attn_b(f_b_tokens)+ f_b_tokens
        # f_c_spec = self.self_attn_c(f_c_tokens) + f_c_tokens

        f_b_spec = self.token_to_spatial(f_b_spec)
        f_c_spec = self.token_to_spatial(f_c_spec)
        # f_invar_c2b = self.token_to_spatial(f_invar_c2b)
        # f_invar_b2c = self.token_to_spatial(f_invar_b2c)        
        # f_invar = self.token_to_spatial(f_invar)

        # f_b_spec_main = self.CIA1(f_b_spec, f_c_spec)
        # f_c_spec_main = self.CIA2(f_b_spec, f_c_spec)
        # LGU
        f_b_spec = self.lgu1(f_b_spec)
        f_c_spec = self.lgu2(f_c_spec)
        f_invar = self.lgu3(f_invar)
        
        # b_fused_features = torch.cat([f_b_spec, f_invar_c2b], dim=1)
        # c_fused_features = torch.cat([f_c_spec, f_invar_b2c], dim=1)

        f_b_spec = self.spatial_to_token(f_b_spec)
        f_c_spec = self.spatial_to_token(f_c_spec)
        f_invar = self.spatial_to_token(f_invar)
        # Concatenate all features
        # Eq(4): F_AMF = Concat(F_B_spec, F_invar, F_C_spec)
        f_concat = torch.cat([f_b_spec, f_invar, f_c_spec], dim=-1)  # [B, H*W, 3*C]
        # f_concat = self.spatial_to_token(f_concat)
        # print(f_invar_b2c.size(), f_invar_c2b.size(), f_b_spec.size(), f_c_spec.size())
        # Final MLP fusion
        fused_tokens = self.mlp(f_concat)
        fused_tokens = self.norm3(fused_tokens)
        
        # Convert back to spatial features
        fused_features = self.token_to_spatial(fused_tokens)  # [B, C, H, W]

        return fused_features


class MultiScaleAMF(nn.Module):
    """
    Multi-scale AMF for handling features at different resolutions
    This is useful when integrating with YOLOX which has multiple feature scales
    """
    def __init__(self, channels_list, feature_sizes, num_heads=8, **kwargs):
        """
        Args:
            channels_list: list of channel dimensions for each scale
            feature_sizes: list of (H, W) tuples for each scale
            num_heads: number of attention heads
        """
        super().__init__()
        self.amf_layers = nn.ModuleList()
        
        for channels, (H, W) in zip(channels_list, feature_sizes):
            amf_layer = AMF(
                dim=channels,
                H=H,
                W=W,
                num_heads=num_heads,
                **kwargs
            )
            self.amf_layers.append(amf_layer)
    
    def forward(self, b_mode_features, ceus_mode_features):
        """
        Args:
            b_mode_features: list of B-mode features at different scales, each [B, C, H, W]
            ceus_mode_features: list of CEUS-mode features at different scales, each [B, C, H, W]
        Returns:
            fused_features: list of fused features at different scales
        """
        fused_features = []
        for amf_layer, f_b, f_c in zip(self.amf_layers, b_mode_features, ceus_mode_features):
            fused_feat = amf_layer(f_b, f_c)
            fused_features.append(fused_feat)
        
        return fused_features


# Example usage and test
if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    batch_size = 2
    b_features = [
        torch.randn(batch_size, 128, 96, 96).cuda(),
        torch.randn(batch_size, 256, 48, 48).cuda(),
        torch.randn(batch_size, 512, 24, 24).cuda()
    ]
    c_features = [
        torch.randn(batch_size, 128, 96, 96).cuda(),
        torch.randn(batch_size, 256, 48, 48).cuda(),
        torch.randn(batch_size, 512, 24, 24).cuda()
    ]
    
    # 检查初始显存
    initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    
    print(f"初始显存占用: {initial_memory:.2f} MB")
    
    # 计算特征图总大小
    total_elements = 0
    for feat_list in [b_features, c_features]:
        for feat in feat_list:
            total_elements += feat.numel()
    
    feature_memory = total_elements * 4 / 1024**2  # FP32
    print(f"特征图理论显存: {feature_memory:.2f} MB")