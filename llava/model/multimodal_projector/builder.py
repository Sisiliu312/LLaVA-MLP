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
    def __init__(self, text_dim, feature_dim):
        super(CrossAttention, self).__init__()
        # 初始化线性变换层
        self.text_dim = text_dim
        self.feature_dim = feature_dim
        self.W_q = nn.Linear(text_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        
        # 初始化参数
        self._reset_parameters()
        print("-=-=-=-=-=-初始化-=-=-=-=-=-=-=-")
        print('[CA]self.W_q :', self.W_q.weight.shape)
        print('[CA]self.W_q.bias :', self.W_k.bias.shape)

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
        # print("-=-=-=-=-=-fordward时-=-=-=-=-=-=-=-")
        # print('fordward时 self.W_q mean:', self.W_q.weight.norm().item())
        print('[CA]self.W_q :', self.W_q.weight.shape)
        print('[CA]self.W_q.bias :', self.W_k.bias.shape)

        # 1. 线性变换
        Q = self.W_q(text)  # [batch, text_seq_len, feature_dim]
        K = self.W_k(features)  # [batch, num_patches, feature_dim]
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

# class LayerSelectionRouter(nn.Module):
#     """
#     Router that selects top-5 layers from 24 vision tower layers.
#     Initialized to uniformly select layers [1, 6, 11, 16, 21] (0-indexed: [0, 5, 10, 15, 20])
#     """
#     def __init__(self, dim, num_layers, top_router):
#         super(LayerSelectionRouter, self).__init__()
#         self.num_layers = num_layers
#         self.top_router = top_router
#         self.dim = dim

#         # print(f"Initializing LayerSelectionRouter with dim={self.dim}, num_layers={self.num_layers}, top_router={self.top_router}")
        
#         # Router network with SiLU gating (matching diagram: W1, W2, W3)
#         self.w1 = nn.Linear(dim, dim)
#         self.w2 = nn.Linear(dim, dim)
#         self.w3 = nn.Linear(dim, num_layers)
        
#         # Initialize to favor uniform selection of [0, 5, 10, 15, 20]
#         self._reset_parameters()
#         # print('初始化时 self.w1 mean:', self.w1.weight.norm().item())
#         print('初始化时 self.w1 shape:', self.w1.weight.shape)
#         # print("初始化时 w3.bias (前6 + 后1):", self.w3.bias.tolist()[:6] + ["..."] + [self.w3.bias.tolist()[-1]])
        
#     def _reset_parameters(self):
#         """Initialize router to uniformly select layers [1, 6, 11, 16, 21] (0-indexed)"""
        
#         # checkpoint = torch.load('/hy-tmp/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin')
#         # print("保存的参数:")
#         # for key in checkpoint.keys():
#         #     print(f"  {key}: {checkpoint[key].shape}")

#         nn.init.xavier_uniform_(self.w1.weight)
#         nn.init.xavier_uniform_(self.w2.weight)
#         nn.init.xavier_uniform_(self.w3.weight)
#         nn.init.constant_(self.w1.bias, 0.0)
#         nn.init.constant_(self.w2.bias, 0.0)
#         nn.init.constant_(self.w3.bias, -0.1)
        
#         with torch.no_grad():
#             uniform_indices = [3, 8, 13, 18, 23]
#             for idx in uniform_indices:
#                 self.w3.bias[idx] = 0.1
                    
    
#     def forward(self, text_features):
#         """
#         Args:
#             text_features: Text token features [batch_size, text_len, dim]
        
#         Returns:
#             layer_weights: Softmax weights for all layers [batch_size, num_layers]
#             selected_indices: Indices of top-k selected layers [batch_size, top_router]
#         """
#         # [batch_size, text_len, dim] -> [batch_size, dim]
#         pooled = text_features.mean(dim=1)
#         # self.attention_pool = nn.Sequential(
#         #     nn.Linear(dim, 1),
#         #     nn.Softmax(dim=1)
#         # )
        
#         # print('fordward时 self.w1 mean:', self.w1.weight.norm().item())
#         print('fordward时 self.w1 shape:', self.w1.weight.shape)
#         h1 = F.silu(self.w1(pooled))
        
#         h2 = F.silu(self.w2(pooled))
        
#         gated = h1 * h2
        
#         # print("forward时 w3.bias (前6 + 后1):", self.w3.bias.tolist()[:6] + ["..."] + [self.w3.bias.tolist()[-1]])
#         logits = self.w3(gated)
        
#         layer_probs = F.softmax(logits, dim=-1)


#         top_weights, top_indices = torch.topk(layer_probs, self.top_router, dim=-1)

#         top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

#         # 向量化投票: 统计哪些层被选择最多
#         # batch_size, text_len, dim = text_features.shape
#         # print("text_features shape:", text_features.shape)
#         # h1 = F.silu(self.w1(text_features))
#         # h2 = F.silu(self.w2(text_features))
#         # print("h2 shape:", h2.shape)
#         # gated = h1 * h2
#         # # [batch_size, text_len, num_layers]
#         # logits = self.w3(gated)  
#         # layer_probs = F.softmax(logits, dim=-1)
#         # print("layer_probs shape:", layer_probs.shape)
#         # per_token_top = torch.topk(layer_probs, self.top_router, dim=-1)

#         # layer_votes = torch.zeros(
#         #     batch_size, 
#         #     self.num_layers, 
#         #     device=text_features.device,
#         #     dtype=text_features.dtype
#         # )
        
#         # for b in range(batch_size):
#         #     layer_votes[b].scatter_add_(
#         #         0,
#         #         per_token_top.indices[b].flatten(),
#         #         per_token_top.values[b].flatten()
#         #     )

#         # top_weights, top_indices = torch.topk(layer_votes, self.top_router, dim=-1)
#         # print("top_weights shape:", top_weights.shape)
        
#         # top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
#         return top_indices, top_weights, layer_probs


class LayerSelectionRouter(nn.Module):
    """
    Router that selects top-5 layers from 24 vision tower layers.
    Initialized to uniformly select layers [1, 6, 11, 16, 21] (0-indexed: [0, 5, 10, 15, 20])
    """
    def __init__(self, dim, num_layers, top_router):
        super(LayerSelectionRouter, self).__init__()
        self.num_layers = num_layers
        self.top_router = top_router
        self.dim = dim
        
        # ✅ 简化的 Router 架构
        self.router = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_layers)
        )
        
        # Initialize to favor uniform selection
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize router to uniformly select layers [1, 6, 11, 16, 21] (0-indexed)"""
        for module in self.router.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        uniform_indices = [3, 8, 13, 18, 23]
        with torch.no_grad():
            for idx in uniform_indices:
                self.router[-1].bias[idx] = 0.5

        print(f"[Router]router[0].weight.shape: {self.router[0].weight.shape}")
        print(f"[Router]router[2].weight.shape: {self.router[2].weight.shape}")
        print(f"[Router]router[2].bias (前6项): {self.router[2].bias[:6].tolist()}")
        print(f"[Router]router[2].bias (后6项): {self.router[2].bias[-6:].tolist()}")
    
    def forward(self, text_features):
        batch_size, text_len, dim = text_features.shape
        
        # 每个 token 都预测
        logits = self.router(text_features)  # [batch_size, text_len, num_layers]
        print(f"[Router]router[0].weight.shape: {self.router[0].weight.shape}")
        print(f"[Router]router[2].weight.shape: {self.router[2].weight.shape}")
        print(f"[Router]router[2].bias (前6项): {self.router[2].bias[:6].tolist()}")
        print(f"[Router]router[2].bias (后6项): {self.router[2].bias[-6:].tolist()}")

        layer_probs = F.softmax(logits, dim=-1)
        layer_probs = layer_probs[0]  # [text_len, num_layers]
        
        # 投票: 聚合所有 token 的预测
        aggregated_probs = layer_probs.mean(dim=0)  # [num_layers]
        
        top_weights, top_indices = torch.topk(aggregated_probs, self.top_router, dim=-1)

        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        return top_indices, top_weights, aggregated_probs

class SimplifiedFeatureFusion(nn.Module):
    def __init__(self, dim, num_layers=5):
        super().__init__()
        
        # 每层特征先过一个小型adapter
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        
        # 跨层融合使用简单MLP
        self.fusion = nn.Sequential(
            nn.Linear(dim * num_layers, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        self._reset_parameters()
        print(f"[Fusion]fusion[0].weight.shape: {self.fusion[0].weight.shape}")
        print(f"[Fusion]fusion[2].weight.shape: {self.fusion[2].weight.shape}")
    
    def _reset_parameters(self):
        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, layer_features_list, weights):
        # 1. 每层归一化后应用权重
        weighted_feats = []
        for feat, norm, w in zip(layer_features_list, self.layer_norms, weights):
            weighted_feats.append(norm(feat) * w)
        
        # 2. 拼接后通过MLP融合
        concat = torch.cat(weighted_feats, dim=-1)
        fused = self.fusion(concat)
        print(f"  [Fusion]fusion[0].weight.shape: {self.fusion[0].weight.shape}")
        print(f"  [Fusion]fusion[2].weight.shape: {self.fusion[2].weight.shape}")
        
        return fused

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

def build_cross_attn(config):
    text_dim = config.text_dim if hasattr(config, 'text_dim') else 4096
    feature_dim = config.feature_dim if hasattr(config, 'feature_dim') else 4096
    
    return CrossAttention(text_dim=text_dim, feature_dim=feature_dim)

def build_layer_router(config):
    # print("=" * 60)
    # print("🔍 build_layer_router 调试:")
    # print(f"  hasattr(config, 'dim'): {hasattr(config, 'dim')}")
    # print(f"  hasattr(config, 'num_layers'): {hasattr(config, 'num_layers')}")
    # print(f"  hasattr(config, 'top_router'): {hasattr(config, 'top_router')}")
    
    dim = config.dim if hasattr(config, 'dim') else 4096
    num_layers = config.num_layers if hasattr(config, 'num_layers') else 24
    top_router = config.top_router if hasattr(config, 'top_router') else 5
    
    # print(f"  最终: dim={dim}, num_layers={num_layers}, top_router={top_router}")
    # print(f"  类型: dim={type(dim)}, num_layers={type(num_layers)}, top_router={type(top_router)}")
    # print("=" * 60)
    
    return LayerSelectionRouter(dim=dim, num_layers=num_layers, top_router=top_router)


def build_layer_fusion(config):
    dim = config.dim if hasattr(config, 'dim') else 4096
    num_layers = config.num_layers if hasattr(config, 'num_layers') else 5
    
    return SimplifiedFeatureFusion(dim=dim, num_layers=num_layers)