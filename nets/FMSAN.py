import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from model.AMF import AMF

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class HMAA(nn.Module):
    def __init__(self, channels, block_size=4, grid_size=4, reduction_ratio=16):
        super(HMAA, self).__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        self.channels = channels
        
        # Channel attention components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_dense_one = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True)
        )
        self.shared_dense_two = nn.Sequential(
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(channels)
        
        # Linear layers for block and grid attention
        self.block_linear = nn.Linear(channels, channels)
        self.grid_linear = nn.Linear(channels, channels)
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Channel attention
        avg_out = self.shared_dense_two(self.shared_dense_one(self.avg_pool(x).view(B, C)))
        max_out = self.shared_dense_two(self.shared_dense_one(self.max_pool(x).view(B, C)))
        channel_att = self.sigmoid(avg_out + max_out).view(B, C, 1, 1)
        channel_refined = x * channel_att
        
        # Height-wise (block) attention
        block_h = channel_refined.view(B, H // self.block_size, self.block_size, W, C)
        block_h = block_h.permute(0, 1, 3, 2, 4).contiguous().view(-1, self.block_size, C)
        block_h = self.layer_norm(block_h)
        block_h_att = torch.sigmoid(self.block_linear(block_h))
        block_h = block_h * block_h_att
        block_h = block_h.view(B, H // self.block_size, W, self.block_size, C)
        block_h = block_h.permute(0, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        
        # Width-wise (grid) attention
        grid_w = channel_refined.view(B, H, W // self.grid_size, self.grid_size, C)
        grid_w = grid_w.permute(0, 2, 1, 3, 4).contiguous().view(-1, self.grid_size, C)
        grid_w = self.layer_norm(grid_w)
        grid_w_att = torch.sigmoid(self.grid_linear(grid_w))
        grid_w = grid_w * grid_w_att
        grid_w = grid_w.view(B, W // self.grid_size, H, self.grid_size, C)
        grid_w = grid_w.permute(0, 2, 1, 3, 4).contiguous().view(B, C, H, W)
        
        # Combine height and width attention
        height_width_att = self.sigmoid(block_h + grid_w)
        spatial_refined = channel_refined * height_width_att
        
        # Enhanced spatial attention
        avg_out = torch.mean(spatial_refined, dim=1, keepdim=True)
        max_out, _ = torch.max(spatial_refined, dim=1, keepdim=True)
        spatial_att_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_att_input))
        
        combined_spatial_att = height_width_att * spatial_att
        refined_output = spatial_refined * combined_spatial_att
        
        return refined_output

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.size()
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class MCAU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MCAU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Multi-scale depthwise convolutions
        self.conv_3x1 = nn.Conv2d(in_channels, in_channels, (3, 1), padding=(1, 0), groups=in_channels, bias=False)
        self.conv_1x3 = nn.Conv2d(in_channels, in_channels, (1, 3), padding=(0, 1), groups=in_channels, bias=False)
        
        self.conv_5x1 = nn.Conv2d(in_channels, in_channels, (5, 1), padding=(2, 0), groups=in_channels, bias=False)
        self.conv_1x5 = nn.Conv2d(in_channels, in_channels, (1, 5), padding=(0, 2), groups=in_channels, bias=False)
        
        self.conv_11x1 = nn.Conv2d(in_channels, in_channels, (11, 1), padding=(5, 0), groups=in_channels, bias=False)
        self.conv_1x11 = nn.Conv2d(in_channels, in_channels, (1, 11), padding=(0, 5), groups=in_channels, bias=False)
        
        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.layer_norm2 = nn.LayerNorm(in_channels)
        self.layer_norm3 = nn.LayerNorm(in_channels)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.layer_norm_mixed = nn.LayerNorm(out_channels)
        
        self.eca = ECALayer(out_channels)
        
    def forward(self, x):
        # Multi-scale convolutions
        conv1 = self.conv_1x3(self.conv_3x1(x))
        conv1 = self.layer_norm1(conv1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        conv2 = self.conv_1x5(self.conv_5x1(x))
        conv2 = self.layer_norm2(conv2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        conv3 = self.conv_1x11(self.conv_11x1(x))
        conv3 = self.layer_norm3(conv3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Combine multi-scale features
        combined = conv1 + conv2 + conv3
        mixed = self.pointwise(combined)
        mixed = self.layer_norm_mixed(mixed.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Apply ECA attention
        eca_att = self.eca(mixed)
        
        # Final output with attention
        output = x * eca_att
        
        return output

class FuzzyLearningModule(nn.Module):
    def __init__(self, in_channels, num_membership_functions=5, uncertainty_drop_rate=0.2):
        super(FuzzyLearningModule, self).__init__()
        self.num_membership_functions = num_membership_functions
        self.uncertainty_drop_rate = uncertainty_drop_rate
        self.in_channels = in_channels
        
        # Learnable Gaussian parameters - 修正维度问题
        self.mu = nn.Parameter(torch.randn(num_membership_functions, in_channels))
        self.sigma = nn.Parameter(torch.randn(num_membership_functions, in_channels))
        
        # Convolution layers
        self.fuzzy_conv = nn.Conv2d(num_membership_functions, in_channels, 1, bias=False)
        self.dropout = nn.Dropout2d(uncertainty_drop_rate)
        
        # High-confidence processing
        self.high_conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.high_dwconv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.high_conv2 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
        # Low-confidence processing
        self.low_dwconv1 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=False)
        self.low_conv1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.low_bn1 = nn.BatchNorm2d(in_channels)
        
        self.low_dwconv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.low_conv2 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
        # Final processing
        self.combined_dwconv = nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1, groups=in_channels * 2, bias=False)
        self.combined_conv = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
        self.final_bn = nn.BatchNorm2d(in_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # 修正模糊成员值计算 - 确保维度匹配
        # x shape: [B, C, H, W]
        # 我们想要为每个通道计算多个成员函数值
        
        # 重新塑形以进行计算
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
        
        # 扩展mu和sigma以匹配批量维度
        mu_expanded = self.mu.unsqueeze(0)  # [1, N, C]
        sigma_expanded = self.sigma.unsqueeze(0)  # [1, N, C]
        
        # 计算模糊成员值
        x_flat_expanded = x_flat.unsqueeze(1)  # [B*H*W, 1, C]
        fuzzy_values = torch.exp(-(x_flat_expanded - mu_expanded)**2 / (2 * sigma_expanded**2 + 1e-8))
        
        # 聚合成员函数值（乘积）
        fuzzy_values = torch.prod(fuzzy_values, dim=2, keepdim=False)  # [B*H*W, N]
        
        # 重新塑形回空间维度
        fuzzy_values = fuzzy_values.view(B, H, W, self.num_membership_functions).permute(0, 3, 1, 2)  # [B, N, H, W]
        
        # Transform fuzzy features
        fuzzy_transformed = self.fuzzy_conv(fuzzy_values)  # [B, C, H, W]
        
        # Monte Carlo Dropout for uncertainty estimation
        # if self.training:
        mc_dropout = self.dropout(fuzzy_transformed)
        # else:
            # mc_dropout = fuzzy_transformed
            
        # 计算不确定性图 - 修正维度处理
        uncertainty_map = torch.var(mc_dropout, dim=1, keepdim=True)  # [B, 1, H, W]
        uncertainty_map = torch.sigmoid(uncertainty_map)
        
        # Separate high and low confidence features
        high_conf_features = x * (1 - uncertainty_map)
        low_conf_features = x * uncertainty_map
        
        # Process high-confidence features
        high_processed = self.high_conv1(high_conf_features)
        high_processed = self.high_dwconv(high_processed)
        high_processed = self.relu(high_processed)
        high_processed = self.high_conv2(high_processed)
        high_processed = self.relu(high_processed)
        
        # Process low-confidence features
        low_processed = self.low_dwconv1(low_conf_features)
        low_processed = self.relu(low_processed)
        low_processed = self.low_conv1(low_processed)
        low_processed = self.low_bn1(low_processed)
        low_processed = self.relu(low_processed)
        
        low_processed = self.low_dwconv2(low_processed)
        low_processed = self.relu(low_processed)
        low_processed = self.low_conv2(low_processed)
        low_processed = self.relu(low_processed)
        
        # Combine features
        combined_features = torch.cat([high_processed, low_processed], dim=1)
        combined_features = self.combined_dwconv(combined_features)
        combined_features = self.relu(combined_features)
        combined_features = self.combined_conv(combined_features)
        combined_features = self.final_bn(combined_features)
        combined_features = self.relu(combined_features)
        
        # Residual connection
        output = x + combined_features
        
        return output, uncertainty_map

class FMSANStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_membership_functions=5):
        super(FMSANStage, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Feed Forward Network
        self.ffn_conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.ffn_ln1 = nn.LayerNorm(out_channels)
        self.ffn_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.ffn_ln2 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Multi-scale Contextual Attention Unit
        self.mcau = MCAU(out_channels, out_channels)
        self.mcau_ln = nn.LayerNorm(out_channels)
        
        # Fuzzy Learning Module
        self.fuzzy_module = FuzzyLearningModule(out_channels, num_membership_functions)
        
        # Downsample
        self.downsample = nn.Conv2d(out_channels, out_channels * 2, 3, stride=2, padding=1, bias=False)
        
    def forward(self, x):
        # FFN
        residual = x
        if x.size(1) != self.out_channels:
            residual = self.ffn_conv1(residual)
            
        ffn_out = self.ffn_conv1(x)
        ffn_out = self.ffn_ln1(ffn_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        ffn_out = self.relu(ffn_out)
        ffn_out = self.ffn_conv2(ffn_out)
        ffn_out = self.ffn_ln2(ffn_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        ffn_out = ffn_out + residual
        
        # MCAU
        mcau_out = self.mcau(ffn_out)
        mcau_out = self.mcau_ln(mcau_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Fuzzy Learning Module
        fuzzy_out, uncertainty_map = self.fuzzy_module(mcau_out)
        
        # Combine with residual
        output = ffn_out + fuzzy_out
        
        # Downsample
        output = self.downsample(output)
        
        return output

class FMSAN(nn.Module):
    def __init__(self, in_channels=3, initial_filters=64, num_stages=4, num_membership_functions=5):
        super(FMSAN, self).__init__()
        self.num_stages = num_stages
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, initial_filters, 3, padding=1, bias=False)
        
        # Stages
        self.stages = nn.ModuleList()
        current_channels = initial_filters
        
        for i in range(num_stages):
            stage = FMSANStage(current_channels, current_channels, num_membership_functions)
            self.stages.append(stage)
            current_channels *= 2
            
    def forward(self, x):
        features = []
        x = self.initial_conv(x)
        features.append(x)
        
        for stage in self.stages:
            x = stage(x)
            features.append(x)
            
        return features
        
class multi_FMSAN(nn.Module):
    def __init__(self, in_channels=3, initial_filters=64, num_stages=4, num_membership_functions=5, fuse_layers=[0,1,2]):
        super(multi_FMSAN, self).__init__()
        self.num_stages = num_stages
        self.fuse_layer = fuse_layers
        # Initial convolution
        self.initial_conv1 = nn.Conv2d(in_channels, initial_filters, 4, 2, padding=1, bias=False)
        self.initial_conv2 = nn.Conv2d(in_channels, initial_filters, 4, 2, padding=1, bias=False)
        
        # Stages
        self.stages1 = nn.ModuleList()
        self.stages2 = nn.ModuleList()
        self.fuse = nn.ModuleList()
        current_channels = initial_filters
        
        for i in range(num_stages):
            stage1 = FMSANStage(current_channels, current_channels, num_membership_functions)
            stage2 = FMSANStage(current_channels, current_channels, num_membership_functions)

            self.stages1.append(stage1)
            self.stages2.append(stage2)
            current_channels *= 2

        
        for i in fuse_layers:
            fuse = AMF(initial_filters * 2 ** (i+1), int(96/(2**i)), int(96/(2**i)), num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mlp_ratio=4.0)

            self.fuse.append(fuse)

            
    def forward(self, x1, x2):
        features = []
        
        x1 = self.initial_conv1(x1)
        x2 = self.initial_conv2(x2)
        
        # features.append(x)
        t = 0
        for i in range(self.num_stages):
            # print(i)
            x1 = self.stages1[i](x1)
            x2 = self.stages2[i](x2)
            # features.append(x1+x2)
            if i in self.fuse_layer:
                # print(x1.size())
                fused_feat = self.fuse[t](x1, x2)
                features.append(fused_feat)
                x1 += fused_feat
                t += 1
            
        return features
        
class WMFM(nn.Module):
    def __init__(self):
        super(WMFM, self).__init__()
        
    def forward(self, encoder_outputs):
        # Transform all features to same spatial size (smallest)
        target_size = encoder_outputs[0].shape[2:]
        transformed_features = []
        single_feature = []
        
        for feature in encoder_outputs:
            if feature.shape[2:] != target_size:
                feature_resized = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            else:
                feature_resized = feature
                
            # Channel adjustment
            if feature_resized.size(1) != encoder_outputs[-1].size(1):
                conv = nn.Conv2d(feature_resized.size(1), encoder_outputs[-1].size(1), 1).to(feature_resized.device)
                feature_resized = conv(feature_resized)
                
            transformed_features.append(feature_resized)
        
        # Learnable weights
        weights = nn.Parameter(torch.ones(len(transformed_features))).to(encoder_outputs[0].device)
        normalized_weights = F.softmax(weights, dim=0)
        
        # Weighted multiplication fusion
        for i in range(len(transformed_features)):
            single_feature.append(transformed_features[i] * normalized_weights[i])
        fused_feature = single_feature[0]
        for i in range(1, len(transformed_features)):
            fused_feature = fused_feature * single_feature[i]
            
        return single_feature, fused_feature

class GMFAM(nn.Module):
    def __init__(self, channels):
        super(GMFAM, self).__init__()
        
        # Shared depthwise separable convolutions
        self.conv1x1 = DepthwiseSeparableConv(channels * 3, channels, 1)
        self.conv3x3 = DepthwiseSeparableConv(channels * 3, channels, 3, padding=1)
        self.conv5x5 = DepthwiseSeparableConv(channels * 3, channels, 5, padding=2)
        
        # Multi-scale fusion
        self.multi_scale_conv = nn.Conv2d(channels * 3, channels * 4, 3, padding=1, groups=channels, bias=False)
        self.multi_scale_bn = nn.BatchNorm2d(channels * 4)
        
        # Gating convolutions
        self.gate_conv4 = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate_conv3 = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate_conv_hmaa = nn.Conv2d(channels, channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
        # Final fusion
        self.final_conv = nn.Conv2d(channels * 4, channels, 1, bias=False)
        
    def forward(self, F4, F3, F_hmaa):
        # Resize features to match F4 spatial dimensions
        target_size = F4.shape[2:]
        F3_resized = F.interpolate(F3, size=target_size, mode='bilinear', align_corners=False)
        F_hmaa_resized = F.interpolate(F_hmaa, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate features
        concatenated = torch.cat([F4, F3_resized, F_hmaa_resized], dim=1)
        
        # Multi-scale feature extraction
        scale1 = self.conv1x1(concatenated)
        scale2 = self.conv3x3(concatenated)
        scale3 = self.conv5x5(concatenated)
        
        multi_scale = torch.cat([scale1, scale2, scale3], dim=1)
        
        # Multi-scale fusion
        Q = self.multi_scale_conv(multi_scale)
        Q = self.multi_scale_bn(Q)
        Q = F.relu(Q)
        
        # Split into components
        split_size = Q.size(1) // 4
        N = Q[:, :split_size]
        g4 = Q[:, split_size:2*split_size]
        g3 = Q[:, 2*split_size:3*split_size]
        g_hmaa = Q[:, 3*split_size:4*split_size]
        
        # Apply gating
        g4_gate = self.sigmoid(self.gate_conv4(g4))
        g3_gate = self.sigmoid(self.gate_conv3(g3))
        g_hmaa_gate = self.sigmoid(self.gate_conv_hmaa(g_hmaa))
        
        F4_gated = F4 * g4_gate
        F3_gated = F3_resized * g3_gate
        F_hmaa_gated = F_hmaa_resized * g_hmaa_gate
        
        # Final fusion
        gated_concat = torch.cat([N, F4_gated, F3_gated, F_hmaa_gated], dim=1)
        output = self.final_conv(gated_concat)
        
        return output

class AAFM(nn.Module):
    def __init__(self, channels):
        super(AAFM, self).__init__()
        
        # Atrous/dilated convolutions with different rates
        self.dconv1 = DepthwiseSeparableConv(channels, channels, 3, dilation=1, padding=1)
        self.dconv2 = DepthwiseSeparableConv(channels, channels, 3, dilation=2, padding=2)
        self.dconv3 = DepthwiseSeparableConv(channels, channels, 3, dilation=4, padding=4)
        self.dconv4 = DepthwiseSeparableConv(channels, channels, 3, dilation=6, padding=6)
        
        self.eca = ECALayer(channels * 4)
        
    def forward(self, x):
        # Apply atrous convolutions
        F1 = self.dconv1(x)
        F1_sum = x + F1
        
        F2 = self.dconv2(F1_sum)
        F2_sum = F1_sum + F2
        
        F3 = self.dconv3(F2_sum)
        F3_sum = F2_sum + F3
        
        F4 = self.dconv4(F3_sum)
        F4_sum = F3_sum + F4
        
        # Concatenate multi-scale features
        concat_features = torch.cat([F1_sum, F2_sum, F3_sum, F4_sum], dim=1)
        
        # Apply ECA attention
        output = self.eca(concat_features)
        
        return output

class GMSFB(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels):
        super(GMSFB, self).__init__()
        
        # Gating mechanism
        self.gating_conv = nn.Conv2d(decoder_channels, out_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # Multi-scale dilated convolutions
        self.dconv1 = DepthwiseSeparableConv(out_channels, out_channels, 3, dilation=1, padding=1)
        self.dconv2 = DepthwiseSeparableConv(out_channels, out_channels, 3, dilation=2, padding=2)
        self.dconv3 = DepthwiseSeparableConv(out_channels, out_channels, 3, dilation=4, padding=4)
        
        # Fusion
        self.fusion_conv = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)
        
    def forward(self, encoder_features, decoder_features):
        # Resize encoder features to match decoder spatial dimensions
        encoder_resized = F.interpolate(encoder_features, size=decoder_features.shape[2:], 
                                      mode='bilinear', align_corners=False)
        
        # Gating mechanism
        gating = self.sigmoid(self.gating_conv(decoder_features))
        attention_features = encoder_resized * gating
        
        # Multi-scale feature extraction
        f1 = self.dconv1(attention_features)
        f2 = self.dconv2(attention_features)
        f3 = self.dconv3(attention_features)
        
        # Concatenate and fuse
        concat_features = torch.cat([f1, f2, f3], dim=1)
        fused_features = self.fusion_conv(concat_features)
        
        # Residual connection
        output = fused_features + attention_features
        
        return output

class FusionSegNet(nn.Module):
    def __init__(self, input_channels=3, input_size=224, num_classes=1):
        super(FusionSegNet, self).__init__()
        
        # Encoder
        self.encoder = FMSAN(in_channels=input_channels, initial_filters=64, num_stages=4)
        
        # Feature fusion modules
        self.wmfm = WMFM()
        self.hmaa = HMAA(512)  # Assuming final encoder channels is 512
        
        # Bottleneck modules
        self.gmfam = GMFAM(512)
        self.aafm = AAFM(512)
        
        # Decoder
        decoder_channels = [512, 256, 128, 64]
        encoder_channels = [512, 256, 128, 64]  # Reverse order for skip connections
        
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.gmsfb_blocks = nn.ModuleList()
        self.hmaa_blocks = nn.ModuleList()
        
        for i in range(len(decoder_channels)):
            # Upsampling
            self.upsample_layers.append(nn.ConvTranspose2d(
                decoder_channels[i] if i == 0 else decoder_channels[i-1],
                decoder_channels[i],
                kernel_size=2,
                stride=2
            ))
            
            # Gated multi-scale fusion block
            self.gmsfb_blocks.append(GMSFB(
                encoder_channels[i],
                decoder_channels[i],
                decoder_channels[i]
            ))
            
            # Hierarchical multi-axis attention
            self.hmaa_blocks.append(HMAA(decoder_channels[i]))
        
        # Final output
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        encoder_features = self.encoder(x)
        
        # Feature fusion
        fused_encoder = self.wmfm(encoder_features)
        enhanced_features = self.hmaa(fused_encoder)
        
        # Bottleneck processing
        gmfa_output = self.gmfam(encoder_features[4], encoder_features[3], enhanced_features)
        atrous_output = self.aafm(gmfa_output)
        
        # Decoder
        decoder_output = atrous_output
        decoder_features = []
        
        for i in range(len(self.decoder_blocks)):
            # Upsample
            decoder_output = self.upsample_layers[i](decoder_output)
            
            # Apply convolution after upsampling
            if i > 0:
                conv = nn.Conv2d(decoder_output.size(1), self.decoder_blocks[i].in_channels, 3, padding=1).to(decoder_output.device)
                decoder_output = conv(decoder_output)
                decoder_output = F.relu(decoder_output)
            
            # Gated multi-scale fusion with skip connection
            skip_idx = 3 - i  # Reverse order for skip connections
            decoder_output = self.gmsfb_blocks[i](encoder_features[skip_idx], decoder_output)
            
            # Hierarchical multi-axis attention
            decoder_output = self.hmaa_blocks[i](decoder_output)
            
            decoder_features.append(decoder_output)
        
        # Final output
        output = self.final_conv(decoder_features[-1])
        output = self.sigmoid(output)
        
        # Resize to original input size if needed
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return output

# Create and test the model
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = FusionSegNet(input_channels=3, input_size=224, num_classes=1)
    model = model.to(device)
    
    # Test with sample input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print("Input shape:", x.shape)
    
    with torch.no_grad():
        output = model(x)
        print("Output shape:", output.shape)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")