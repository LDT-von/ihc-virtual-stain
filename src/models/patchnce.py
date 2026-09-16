"""
PatchNCE Loss (Contrastive Unpaired Translation)
论文：Park et al., "Contrastive Learning for Unpaired Image-to-Image Translation" (ECCV 2020)

核心思想：
- 不需要 paired 数据
- 在特征空间做 patch-level 对比学习
- 正样本：G(A) 与 A 在同位置的 patch
- 负样本：同一 batch 内其他 patch

与 paired pix2pix 的区别：
- pix2pix：需要 pixel-level 配对，直接监督
- CUT/PatchNCE：只需要 unpaired 数据，用对比学习约束特征空间
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    """
    PatchNCE 损失
    
    对每个 patch，计算：
    - exp(sim(z_pos, z_neg) / tau) / sum(exp(sim(z_pos, z_neg_i) / tau))
    
    正样本：同位置的 patch
    负样本：batch 内所有其他 patch
    """
    def __init__(self, opt, batch_size, nce_T=0.07):
        super().__init__()
        self.opt = opt
        self.nce_T = nce_T
        self.batch_size = batch_size
        # Feature 维度 (PatchNCE paper 默认 256)
        self.feat_dim = 256
        # 构建 MLP 投影头（见 CUT paper）
        self.netF = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        # 将 MLP 参数移到正确设备（在 forward 时处理）
    
    def forward(self, feat_q, feat_k):
        """
        Args:
            feat_q: query 特征列表，每个 (B, C, H, W)
            feat_k: key 特征列表，每个 (B, C, H, W)
        Returns:
            PatchNCE loss scalar
        """
        # B = batch size, L = H*W (patch 数), C = feature dim
        B, C, H, W = feat_q.shape
        
        # (B, C, H*W) -> (B, H*W, C)
        feat_q = feat_q.view(B, C, -1).permute(0, 2, 1)  # (B, L, C)
        feat_k = feat_k.view(B, C, -1).permute(0, 2, 1)  # (B, L, C)
        
        # MLP 投影
        feat_q = self.netF(feat_q)  # (B, L, C)
        feat_k = self.netF(feat_k)  # (B, L, C)
        
        # 归一化
        feat_q = F.normalize(feat_q, dim=-1)
        feat_k = F.normalize(feat_k, dim=-1)
        
        # 正样本：同位置 patch 的相似度
        # 形状 (B, L)
        pos_sim = torch.bmm(feat_q, feat_k.transpose(-2, -1)).diag()  # (B,)
        
        # 负样本：同一 batch 内所有其他位置
        # 全部相似度矩阵 (B, L, L)
        neg_sim = torch.bmm(feat_q, feat_k.transpose(-2, -1)) / self.nce_T
        # 排除对角线（自己）
        batch_indices = torch.arange(B, device=feat_q.device)
        neg_sim[batch_indices, :, torch.arange(H * W, device=feat_q.device)] = float('-inf')
        
        # log-softmax
        nce_loss = -neg_sim.logsumexp(dim=-1).mean()
        
        # 也可以用 InfoNCE 形式
        # loss = -log(exp(pos_sim/T) / (exp(pos_sim/T) + sum(exp(neg_sim_i/T))))
        
        return nce_loss


class MultiLayerPatchNCELoss(nn.Module):
    """
    在多层特征上同时计算 PatchNCE（与 CUT 论文一致）
    默认在 layer 0,4,8,12,16 计算（ResNet 风格）
    """
    def __init__(self, opt, nce_T=0.07, nce_layers='0,4,8,12,16'):
        super().__init__()
        self.opt = opt
        self.nce_T = nce_T
        self.nce_layers = [int(x) for x in nce_layers.split(',')]
        self.netF = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
            ) for _ in self.nce_layers
        ])
    
    def forward(self, feat_source, feat_target, netF_list=None):
        """
        Args:
            feat_source: 来自源域的特征列表 [layer_0, layer_4, ...]
            feat_target: 来自目标域的特征列表 [layer_0, layer_4, ...]
            netF_list: 可选，外部传入的 netF（用于 CUT 的共享 PatchNCE MLP）
        Returns:
            总 PatchNCE loss
        """
        nce_loss = 0.0
        for idx, layer_idx in enumerate(self.nce_layers):
            f_q = feat_source[layer_idx]   # query: source 特征
            f_k = feat_target[layer_idx]  # key: target 特征
            
            if netF_list is not None:
                netF = netF_list[idx]
            else:
                netF = self.netF[idx]
            
            # 确保 netF 在正确设备
            netF = netF.to(f_q.device)
            
            B, C, H, W = f_q.shape
            L = H * W
            
            # (B, C, H*W) -> (B, H*W, C)
            f_q = f_q.view(B, C, -1).permute(0, 2, 1)
            f_k = f_k.view(B, C, -1).permute(0, 2, 1)
            
            # MLP 投影 + 归一化
            f_q = F.normalize(netF(f_q), dim=-1)
            f_k = F.normalize(netF(f_k), dim=-1)
            
            # 全部 patch 对的相似度 (B, L, L)
            # 同行同列是对角线 = 正样本
            sim = torch.bmm(f_q, f_k.transpose(-2, -1)) / self.nce_T
            
            # 取对角线（正样本）
            batch_indices = torch.arange(B, device=f_q.device)
            diag_indices = torch.arange(L, device=f_q.device)
            pos_sim = sim[batch_indices, diag_indices, diag_indices]  # (B,)
            
            # 负样本：对角线以外的元素，行方向 softmax
            # 让 sim 的每一行减去 row_max（数值稳定）
            row_max = sim.max(dim=-1, keepdim=True).values
            sim_stable = sim - row_max
            log_sum_exp = torch.log(torch.exp(sim_stable).sum(dim=-1) + 1e-8) + row_max.squeeze(-1)
            nce_per_sample = -pos_sim + log_sum_exp[:, diag_indices]
            
            # 排除自己（对角线贡献的 exp(0/T) = 1 要去掉）
            # 实际上 log_sum_exp 已经包含了 exp(0)=1，但我们不要它
            # 所以 loss = -pos_sim + (log_sum_exp - 0)
            # = -pos_sim + log_sum_exp (因为 exp(0)=1 的 log = 0)
            # 等价于 InfoNCE 形式
            
            nce_loss += nce_per_sample.mean()
        
        return nce_loss / len(self.nce_layers)


def poolfeat(feat, num_patches=256, device='cuda'):
    """
    从特征图采样 num_patches 个 patch
    
    Args:
        feat: (B, C, H, W) 特征图
        num_patches: 采样数
        device: 输出设备
    Returns:
        pooled: (B*num_patches, C)
        indices: (B, num_patches) patch 索引
    """
    B, C, H, W = feat.shape
    L = H * W
    
    if L <= num_patches:
        # 直接 reshape
        feat_flat = feat.view(B, C, -1).permute(0, 2, 1)  # (B, L, C)
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        return feat_flat.reshape(-1, C), indices
    
    # 随机采样 patch
    indices = torch.randint(0, L, (B, num_patches), device=device)  # (B, num_patches)
    
    # gather 操作
    feat_flat = feat.view(B, C, -1)  # (B, C, L)
    # indices: (B, num_patches) -> (B, 1, num_patches)
    # bmm: (B, C, L) @ (B, L, num_patches) -> (B, C, num_patches)
    indices_expanded = indices.unsqueeze(1).expand(B, C, num_patches)  # (B, C, num_patches)
    pooled = torch.gather(feat_flat, 2, indices_expanded)  # (B, C, num_patches)
    pooled = pooled.permute(0, 2, 1).reshape(B * num_patches, C)  # (B*num_patches, C)
    
    return pooled, indices
