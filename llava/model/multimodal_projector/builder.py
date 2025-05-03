import os
import numpy as np
import torch
import torch.nn as nn
import re
import torch.nn.functional as F
from datetime import datetime

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    
# class CrossAttention(nn.Module):
#     def __init__(self, text_dim=4096, feature_dim=4096):
#         super(CrossAttention, self).__init__()
#         self.W_q = nn.Linear(text_dim, feature_dim)
#         self.W_k = nn.Linear(feature_dim, feature_dim)

#     def forward(self, text, features):
#         # print("Text input scale:", text.min(), text.max())  
#         # print("Image input scale:", features.min(), features.max())  
#         # text: [batch, text_seq_len, text_dim] 文本特征
#         # features: [batch, num_patches, feature_dim] 图像特征
        
#         # 计算查询向量Q和键向量K
#         Q = self.W_q(text)  # [batch, text_seq_len, feature_dim]
#         K = self.W_k(features)  # [batch, num_patches, feature_dim]
#         # Q = Q / (Q.norm(dim=-1, keepdim=True) + 1e-6)
#         # K = K / (K.norm(dim=-1, keepdim=True) + 1e-6)

#         # print("-------------------q:", self.W_q.weight.data)
#         # print("*******************k:", self.W_k.weight.data)

#         # print("-------------------Q:", Q.data)
#         # print("*******************K:", K.data)
        

#         # print("Q max/min:", Q.max(), Q.min())  
#         # print("K max/min:", K.max(), K.min()) 
#         # 计算注意力分数
#         attn_scores = torch.matmul(Q, K.transpose(1, 2))
#         attn_scores = attn_scores / (K.size(-1) ** 0.5)
#         # print("^^^^^^^^^^^^^^^^^^^^^^^attn_scores为:",attn_scores)
#         attn_weights = torch.softmax(attn_scores, dim=-1)
#         # print("^^^^^^^^^^^^^^^^^^^^^^^attn_weights为:",attn_weights)
        
#         # 在text_seq_len维度上取平均，得到每块的注意力权重
#         attn_weights = attn_weights.mean(dim=1)  # [batch, num_patches]
        
#         return attn_weights
class SafeLayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        return F.layer_norm(
            x.to(torch.float32),  # 先转换为 float32
            self.normalized_shape,
            self.weight.to(torch.float32) if self.weight is not None else None,
            self.bias.to(torch.float32) if self.bias is not None else None,
            self.eps
        ).to(orig_dtype)  # 再转回原始 dtype

class AttentionWeightSaver:
    def __init__(self, save_dir='attention_weights', format='pt'):
        """
        初始化保存器
        :param save_dir: 保存目录路径
        :param format: 保存格式 ('pt' for PyTorch, 'npy' for NumPy)
        """
        self.save_dir = save_dir
        self.format = format
        self.counter = 0
        self._create_save_dir()
        
    def _create_save_dir(self):
        """创建保存目录"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
    def _generate_filename(self):
        """生成按序号递增的文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attn_{self.counter:04d}_{timestamp}"
        self.counter += 1
        return os.path.join(self.save_dir, f"{filename}.{self.format}")
    
    def save(self, attn_weights, metadata=None):
        """
        保存attention weights
        :param attn_weights: 要保存的attention weights张量
        :param metadata: 可选的元数据字典
        """
        filename = self._generate_filename()
        
        if self.format == 'pt':
            # 保存为PyTorch文件
            save_dict = {'attn_weights': attn_weights}
            if metadata:
                save_dict['metadata'] = metadata
            torch.save(save_dict, filename)
        elif self.format == 'npy':
            # 保存为NumPy文件
            if isinstance(attn_weights, torch.Tensor):
                attn_weights = attn_weights.cpu().numpy()
            if metadata:
                np.savez(filename, attn_weights=attn_weights, **metadata)
            else:
                np.save(filename, attn_weights)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
            
        print(f"Saved attention weights to: {filename}")
        return filename
    
saver = AttentionWeightSaver(save_dir='/home/data/shika/LLaVA/playground/data/eval/textvqa', format='pt')

class CrossAttention(nn.Module):
    def __init__(self, text_dim=4096, feature_dim=4096):
        super(CrossAttention, self).__init__()
        # 初始化线性变换层
        self.W_q = nn.Linear(text_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        
        # 添加LayerNorm归一化层
        # self.q_norm = SafeLayerNorm(feature_dim, eps=1e-6)
        # self.k_norm = SafeLayerNorm(feature_dim, eps=1e-6)
        # self.q_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        # self.k_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        
        # 初始化参数
        self._reset_parameters()

    # def _reset_parameters(self):
    #     # 对线性层使用 Kaiming 初始化（更激进）
    #     for p in [self.W_q.weight, self.W_k.weight]:
    #         if p.dim() > 1:
    #             nn.init.kaiming_uniform_(p, a=math.sqrt(5))
    #     # 偏置初始化为 0
    #     for p in [self.W_q.bias, self.W_k.bias]:
    #         if p is not None:
    #             nn.init.constant_(p, 0.0)
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.constant_(self.W_q.bias, 0.0)
        nn.init.constant_(self.W_k.bias, 0.0)

    def forward(self, text, features):
        """
        text: [batch, text_seq_len, text_dim] 文本特征
        features: [batch, num_patches, feature_dim] 图像特征
        返回: [batch, num_patches] 注意力权重
        """

        # 1. 线性变换
        Q = self.W_q(text)  # [batch, text_seq_len, feature_dim]
        K = self.W_k(features)  # [batch, num_patches, feature_dim]
        # print('Q mean:', Q.mean(), 'std:', Q.std())
        # print('K mean:', K.mean(), 'std:', K.std())
        # print(f"K min: {K.min()}, max: {K.max()}")

        # Q = Q.clamp(min=-50, max=50)
        # K = K.clamp(min=-50, max=50)
        
        # # 2. 应用层归一化
        # Q = self.q_norm(Q)
        # K = self.k_norm(K)
        # print('Q mean:', Q.mean(), 'std:', Q.std())
        # print('K mean:', K.mean(), 'std:', K.std())
        # print(f"K min: {K.min()}, max: {K.max()}")
        
        # 3. 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(1, 2))  # [batch, text_seq_len, num_patches]

        # ✅ 注意：这里默认不再进行缩放
        attn_scores = attn_scores / (K.size(-1) ** 0.5)
        # print("attn_scores std:", attn_scores.std().item())
        # print("attn_scores min:", attn_scores.min().item())
        # print("attn_scores max:", attn_scores.max().item())

        # 4. 应用 softmax 获取注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # print("权重为:",attn_weights.mean())
        
        # saver.save(attn_weights)

        attended = torch.matmul(attn_weights, features)  # [B, T, D]

        return attended
        

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_cross_attn():
    return CrossAttention()
