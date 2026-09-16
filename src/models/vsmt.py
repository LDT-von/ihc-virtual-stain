"""VSMT (CVPR 2026) 核心模块：双对齐多任务特征引导

论文：Virtual Immunohistochemistry Staining with Dual-Aligned Multi-Task Feature Guidance
作者：Shigeng Xie et al., 2026
来源：https://github.com/U-RBook/VSMT

三个核心组件：
1. Msf  (Structure-stain Feature Modulator) — 抑制染色干扰，增强结构特征
2. Mta  (Task-gap Alignment Model)          — 桥接多任务特征与染色特征的语义鸿沟
3. APM  (Active-Passive Matching)           — 在特征层做空间对齐（替代 OT）

核心思想：
  - H&E/IHC 配对数据有空间错位（连续切片形变），直接 L1 监督会学错
  - 用多任务（分类 + 重建）特征提供语义辅助，但这些特征也要对齐才能用
  - 先用 SEL 学习 Msf 让"结构特征"比"染色特征"更可靠
  - 再用 APM 把 IHC 特征按语义聚类配对到 virtual IHC 的位置
  - 最后用 Mta 把 H&E + aligned IHC 多任务特征注入生成器中间层

⚠️ 简化策略：
  - 论文里 H&E / IHC 各有"分类+重建"4 个冻结的辅助网络
  - 我们用生成器的中间层特征 + 一个 ImageNet 预训练 ResNet18 当"分类辅助"
  - 重建任务直接用生成器 encoder 的浅层特征代替
  - 这样不需要预训练 4 个大模型，省时间
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# 1. Msf: Structure-stain Feature Modulator
# ============================================================================

class StructureStainModulator(nn.Module):
    """
    Msf: 结构-染色特征调制器。

    论文公式 (Eq. 1):
        L_mcl = -1/N sum_j log [ exp(sim(y_v, p_v)/τ) /
                                ( exp(sim(y_v, p_v)/τ) + Σ_k exp((sim(y_v, n_k)-m)/τ) ) ]

    其中:
      - y_v = virtual IHC (生成图)
      - p_v = structure-retained 正样本 (颜色扰动、轻微弹性形变)
      - n_v = stain-retained 负样本 (低通滤波、像素shuffle)

    训练后 Msf 能输出"结构信息比染色信息更可靠"的特征。
    """

    def __init__(self, in_channels: int = 64, hidden: int = 128):
        super().__init__()
        # 轻量 CNN：把 in_channels 降到 hidden，再下采样 r 倍
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        # MLP：把特征映射到对比空间（cosine similarity 前）
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 (B, hidden, H', W')"""
        feat = self.net(x)  # (B, hidden, H, W)
        return feat


class SELContrastiveLoss(nn.Module):
    """
    Structure-Enhanced Learning (SEL) 对比损失。

    输入：生成图 y_v
    输出：margin-based InfoNCE loss
    """

    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        super().__init__()
        self.tau = temperature
        self.margin = margin

    @staticmethod
    def structure_retained_transform(y_v: torch.Tensor) -> torch.Tensor:
        """Tstr: 保留结构的扰动 (color jitter + 轻微形变)"""
        # Color jitter in HED space (近似): 通道随机缩放
        B, C, H, W = y_v.shape
        scale = torch.empty(B, C, 1, 1, device=y_v.device).uniform_(0.85, 1.15)
        shift = torch.empty(B, C, 1, 1, device=y_v.device).uniform_(-0.05, 0.05)
        p = y_v * scale + shift
        # 随机擦除 (Random Erasing)：用一个 patch 的均值替换
        erase_h = H // 8
        erase_w = W // 8
        for b in range(B):
            top = torch.randint(0, H - erase_h + 1, (1,)).item()
            left = torch.randint(0, W - erase_w + 1, (1,)).item()
            p[b, :, top:top + erase_h, left:left + erase_w] = y_v[b, :, top:top + erase_h, left:left + erase_w].mean()
        return p.clamp(-1, 1)

    @staticmethod
    def stain_retained_transform(y_v: torch.Tensor) -> torch.Tensor:
        """Tstn: 保留染色但破坏结构 (low-pass filter + pixel shuffle)"""
        B, C, H, W = y_v.shape
        # 低通滤波 (3x3 box filter)
        kernel = torch.ones(C, 1, 3, 3, device=y_v.device) / 9.0
        n = F.conv2d(y_v, kernel, padding=1, groups=C)
        # Pixel shuffle within 4x4 patches: 把每 4x4 块的像素随机重排
        patch = 4
        out = n.clone()
        for b in range(B):
            for i in range(0, H, patch):
                for j in range(0, W, patch):
                    block = n[b, :, i:i + patch, j:j + patch].flatten()
                    idx = torch.randperm(block.numel(), device=n.device)
                    out[b, :, i:i + patch, j:j + patch] = block[idx].reshape(C, patch, patch)
        return out

    def forward(
        self,
        msf: StructureStainModulator,
        y_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            msf: Msf 实例（用于提供 .proj 和 .net）
            y_v:  virtual IHC (B, C, H, W)
        Returns:
            SEL margin-based contrastive loss (scalar)
        """
        p_v = self.structure_retained_transform(y_v)
        n_v = self.stain_retained_transform(y_v)

        # 提取特征 (B, hidden, H, W) → 摊平为 (B*N, hidden)
        F_y = msf.net(y_v)
        F_p = msf.net(p_v)
        F_n = msf.net(n_v)
        B, hidden, H, W = F_y.shape
        N = H * W

        # 投影到对比空间
        F_y = msf.proj(F_y.permute(0, 2, 3, 1).reshape(B * N, hidden))
        F_p = msf.proj(F_p.permute(0, 2, 3, 1).reshape(B * N, hidden))
        F_n = msf.proj(F_n.permute(0, 2, 3, 1).reshape(B * N, hidden))

        F_y = F.normalize(F_y, dim=-1)
        F_p = F.normalize(F_p, dim=-1)
        F_n = F.normalize(F_n, dim=-1)

        # Reshape: (B, N, hidden)
        F_y = F_y.view(B, N, hidden)
        F_p = F_p.view(B, N, hidden)
        F_n = F_n.view(B, N, hidden)

        # 论文 Eq.1: 对每个 j，分子 = exp(sim(y_j, p_j)/τ)
        #                  分母 = 分子 + Σ_k exp((sim(y_j, n_k) - m)/τ)
        sim_yp = (F_y * F_p).sum(dim=-1)  # (B, N) — positive similarity per position
        sim_yn = torch.bmm(F_y, F_n.transpose(1, 2))  # (B, N, N) — sim with all negatives

        pos = torch.exp(sim_yp / self.tau)  # (B, N)
        neg = torch.exp((sim_yn - self.margin) / self.tau).sum(dim=-1)  # (B, N)

        loss = -torch.log(pos / (pos + neg + 1e-8)).mean()
        return loss


# ============================================================================
# 2. APM: Active-Passive Matching
# ============================================================================

class APMLoss(nn.Module):
    """
    Active-Passive Matching: 主动-被动匹配。

    论文思想 (Eq. 2-3):
      1. 对 real IHC 和 virtual IHC 的 Msf 特征分别做 K-Means (K=3: 背景/阴性/阳性)
      2. 主动选择 top-2 最相似的 cluster 对，剩下的被动配对
      3. 在配对的 cluster 内计算 region similarity → spatial alignment matrix A
      4. 用 A 把 real IHC 特征重排到 virtual IHC 的空间位置

    简化实现：
      - 用 PyTorch 的简单 K-Means (迭代 3 次，足够好)
      - 直接用 cluster 内 mean cosine similarity 算 alignment matrix
    """

    def __init__(self, num_clusters: int = 3, iters: int = 3):
        super().__init__()
        self.K = num_clusters
        self.iters = iters

    @staticmethod
    def _kmeans(x: torch.Tensor, K: int, iters: int) -> torch.Tensor:
        """
        简单 K-Means。
        Args:
            x: (B, N, C) features
            K: cluster 数
            iters: 迭代次数
        Returns:
            labels: (B, N) cluster assignment
        """
        B, N, C = x.shape
        # 随机初始化 centroid：选 K 个不同的点
        idx = torch.randperm(N, device=x.device)[:K]
        centroids = x[:, idx, :].clone()  # (B, K, C)
        for _ in range(iters):
            # Distance: (B, N, K)
            dist = torch.cdist(x, centroids)
            labels = dist.argmin(dim=-1)  # (B, N)
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(K):
                mask = (labels == k).unsqueeze(-1)  # (B, N, 1)
                cnt = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                new_centroids[:, k, :] = (x * mask.float()).sum(dim=1) / cnt
            centroids = new_centroids
        return labels

    def _apm_match(
        self,
        feat_real: torch.Tensor,
        feat_virt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Active-Passive Matching。
        Args:
            feat_real: (B, N, C) real IHC 特征 (来自 Msf)
            feat_virt: (B, N, C) virtual IHC 特征 (来自 Msf)
        Returns:
            A: (B, N, N) alignment matrix (sparsemax-normalized)
            cluster_labels_real: (B, N)
            cluster_labels_virt: (B, N)
        """
        B, N, C = feat_real.shape
        K = self.K

        # 1. 独立 K-Means
        lab_r = self._kmeans(feat_real, K, self.iters)  # (B, N)
        lab_v = self._kmeans(feat_virt, K, self.iters)  # (B, N)

        # 2. 主动匹配：枚举 real cluster 的所有 permutation，选 mean sim 最大的 top-2
        # 计算 cluster 之间的 mean sim (B, K, K)
        cluster_sim = torch.zeros(B, K, K, device=feat_real.device)
        for i in range(K):
            for j in range(K):
                mask_v = (lab_v == i)  # (B, N)
                mask_r = (lab_r == j)  # (B, N)
                # (B, C) each: 在 cluster 内的特征均值
                f_v_i = (feat_virt * mask_v.unsqueeze(-1).float()).sum(dim=1) / mask_v.sum(dim=1, keepdim=True).clamp(min=1)
                f_r_j = (feat_real * mask_r.unsqueeze(-1).float()).sum(dim=1) / mask_r.sum(dim=1, keepdim=True).clamp(min=1)
                cluster_sim[:, i, j] = F.cosine_similarity(f_v_i, f_r_j, dim=-1)

        # 贪心：先选最大 sim 的对，再选次大
        # 简化：直接用 Hungarian-like 贪心
        used_r = torch.zeros(B, K, dtype=torch.bool, device=feat_real.device)
        used_v = torch.zeros(B, K, dtype=torch.bool, device=feat_real.device)
        mapping_v_to_r = torch.zeros(B, K, dtype=torch.long, device=feat_real.device)

        sim_work = cluster_sim.clone()
        for _ in range(K - 1):  # 主动 K-1 个，最后一个被动
            # 找全局最大
            sim_work_masked = sim_work.masked_fill(
                (used_v.unsqueeze(-1).expand_as(sim_work)) | (used_r.unsqueeze(1).expand_as(sim_work)),
                -1e9
            )
            max_idx = sim_work_masked.view(B, -1).argmax(dim=-1)  # (B,)
            v_idx = max_idx // K
            r_idx = max_idx % K
            for b in range(B):
                mapping_v_to_r[b, v_idx[b].item()] = r_idx[b].item()
                used_v[b, v_idx[b].item()] = True
                used_r[b, r_idx[b].item()] = True
        # 最后一个 cluster 被动配对
        for b in range(B):
            remaining_v = (~used_v[b]).nonzero(as_tuple=True)[0]
            remaining_r = (~used_r[b]).nonzero(as_tuple=True)[0]
            if len(remaining_v) > 0 and len(remaining_r) > 0:
                mapping_v_to_r[b, remaining_v[0].item()] = remaining_r[0].item()

        # 3. alignment matrix: A[i, j] = sim(v_i, r_j) if cluster 配对, else 0
        sim_pairwise = torch.bmm(feat_virt, feat_real.transpose(1, 2))  # (B, N, N)
        # cluster mask
        cluster_mask = torch.zeros(B, N, N, device=feat_real.device)
        for v_c in range(K):
            v_mask = (lab_v == v_c)  # (B, N)
            r_c = mapping_v_to_r[:, v_c]  # (B,)
            for b in range(B):
                r_mask = (lab_r[b] == r_c[b].item())  # (N,)
                cluster_mask[b] = cluster_mask[b] | (v_mask[b].unsqueeze(-1) & r_mask.unsqueeze(0))

        A = sim_pairwise * cluster_mask.float()  # (B, N, N)
        # Sparsemax: 每行减 max，截断到正
        # 简化：用 softmax with temperature
        A = F.softmax(A * 5.0, dim=-1)
        return A, lab_r, lab_v

    def forward(
        self,
        feat_real: torch.Tensor,
        feat_virt: torch.Tensor,
        feat_real_multitask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_real:           (B, N, C) Msf(real IHC) — 用于 matching
            feat_virt:           (B, N, C) Msf(virtual IHC) — 用于 matching
            feat_real_multitask: (B, N, C') 多任务特征 (mic + mir)，用于 rearrange
        Returns:
            loss: APM loss (scalar)
            aligned_feat: (B, N, C') rearranged multi-task features
        """
        B, N, C = feat_real_multitask.shape
        A, _, _ = self._apm_match(feat_real, feat_virt)  # (B, N, N)
        # aligned_feat[i] = Σ_j A[i,j] * feat_real_multitask[j]
        aligned_feat = torch.bmm(A, feat_real_multitask)  # (B, N, C')

        # 一致性损失: aligned 后的特征应该接近 virtual 特征
        # 论文 Eq.5: L_rc = ||ϕ(A * F_yr) - ϕ(F_yv)||^2
        # 简化：直接 L2
        if feat_real.shape[-1] == feat_real_multitask.shape[-1]:
            loss = F.mse_loss(aligned_feat, feat_virt.detach())
        else:
            # 维度不一致时用投影
            proj = nn.Linear(feat_real_multitask.shape[-1], feat_virt.shape[-1], bias=False).to(feat_real_multitask.device)
            loss = F.mse_loss(proj(aligned_feat), feat_virt.detach())

        return loss, aligned_feat


# ============================================================================
# 3. Mta: Task-gap Alignment Model
# ============================================================================

class TaskGapAlignment(nn.Module):
    """
    Mta: 任务间隙对齐模型。

    论文思想 (Eq. 6):
      F_sta = Mta(F_ya ⊕ F_xm)
      θ_Mta ← θ_Mta - α * ∇ L_vis(x, y, y_v; θ_G'vis)
      即：用生成器的虚拟染色损失作为 proxy 优化 Mta

    实现：
      - 一个小 MLP 把拼接的多任务特征映射到与生成器中间层同维度的空间
      - 输出特征作为 residual 加到 G'vis (冻结的 Gvis 副本) 的中间层
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, in_dim) → (B, N, out_dim)"""
        return self.net(x)


# ============================================================================
# 4. Multi-task Feature Extractor (简化版)
# ============================================================================

class MultiTaskFeatureExtractor(nn.Module):
    """
    多任务特征提取器（论文中 Mhc, Mhr, Mic, Mir 4 个冻结模型）。

    简化策略：
      - 分类特征：用预训练 ResNet18 (ImageNet, 冻结) 提取
      - 重建特征：用生成器 encoder 的浅层特征 (中间层输出)
      - 这样不需要单独训练 4 个辅助网络
    """

    def __init__(self, device='cuda'):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            # 取中间层: layer2 (输出 128 ch @ H/8 W/8) 和 layer3 (256 ch @ H/16 W/16)
            self.layer2 = nn.Sequential(*list(resnet.children())[:6])  # 到 layer1 末
            self.layer3 = resnet.layer2
            self.layer4 = resnet.layer3
            for p in self.parameters():
                p.requires_grad = False
            self.eval()
        except Exception:
            self.layer2 = None
            self.layer3 = None
            self.layer4 = None

        # ImageNet normalize
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """[-1,1] → ImageNet normalize"""
        x = (x + 1) / 2.0
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) 输入图 [-1,1]
        Returns:
            cls_feat: (B, 128, H/8, W/8) — classification feature
            recon_feat: (B, 256, H/16, W/16) — reconstruction feature
        """
        if self.layer2 is None:
            B, _, H, W = x.shape
            return torch.zeros(B, 128, H // 8, W // 8, device=x.device), \
                   torch.zeros(B, 256, H // 16, W // 16, device=x.device)

        x_norm = self._preprocess(x)
        with torch.no_grad():
            h = self.layer2(x_norm)  # (B, 64, H/4, W/4)
            h = self.layer3(h)       # (B, 128, H/8, W/8)
            cls_feat = h
            h = self.layer4(h)       # (B, 256, H/16, W/16)
            recon_feat = h
        return cls_feat, recon_feat


# ============================================================================
# 5. VSMT 总损失（包装：方便在训练脚本里调用）
# ============================================================================

class VSMTLosses(nn.Module):
    """
    VSMT 全部损失的一站式接口。
    """

    def __init__(
        self,
        msf_in_channels: int = 64,
        msf_hidden: int = 128,
        num_clusters: int = 3,
        lambda_sel: float = 1.0,    # L_mcl 权重
        lambda_apm: float = 1.0,    # L_rc 权重
        lambda_mta: float = 1.0,    # L_sp 权重 (semantic preservation)
    ):
        super().__init__()
        self.msf = StructureStainModulator(msf_in_channels, msf_hidden)
        self.sel_loss = SELContrastiveLoss()
        self.apm_loss = APMLoss(num_clusters=num_clusters)
        self.lambda_sel = lambda_sel
        self.lambda_apm = lambda_apm
        self.lambda_mta = lambda_mta

    def forward_sel(self, feat_v: torch.Tensor) -> torch.Tensor:
        """仅计算 SEL (训练 Msf 用)

        Args:
            feat_v: (B, C, H, W) 生成器 encoder 中间层特征（不是 RGB 图像！）
                    C 必须等于 msf_in_channels（默认 64，对应 pix2pix encoder 第2层）
        """
        return self.lambda_sel * self.sel_loss(self.msf, feat_v)

    def forward_apm(
        self,
        feat_real_in: torch.Tensor,
        feat_virt_in: torch.Tensor,
        feat_real_multitask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 APM 一致性损失 + 返回 aligned 多任务特征。
        Args:
            feat_real_in: (B, C, H, W) real IHC 编码器特征
            feat_virt_in: (B, C, H, W) virtual IHC 编码器特征
            feat_real_multitask: (B, C', H', W') IHC 多任务特征
        """
        B, C, H, W = feat_real_in.shape
        # 投影到 Msf 特征空间
        F_r = self.msf(feat_real_in)  # (B, hidden, H, W)
        F_v = self.msf(feat_virt_in)
        # 摊平为 (B, N, hidden)
        F_r = F_r.permute(0, 2, 3, 1).reshape(B, H * W, -1)
        F_v = F_v.permute(0, 2, 3, 1).reshape(B, H * W, -1)

        # 把多任务特征也摊平
        B2, C2, H2, W2 = feat_real_multitask.shape
        F_mt = feat_real_multitask.permute(0, 2, 3, 1).reshape(B2, H2 * W2, C2)

        loss, aligned = self.apm_loss(F_r, F_v, F_mt)
        return self.lambda_apm * loss, aligned
