"""SSIM Loss + L1 Loss + Perceptual Loss 用于 Pix2Pix GAN 训练"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def clamp_safe(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """安全 clamp：将 x 限制在 [eps, 1-eps] 范围，防止 log(0) 爆炸"""
    return x.clamp(min=eps, max=1.0 - eps)


class SSIMLoss(nn.Module):
    """
    SSIM (Structural Similarity Index) Loss
    
    直接优化 SSIM 可以显著提升评测指标分数
    """
    def __init__(self, window_size: int = 11, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', self._create_window(window_size, channel))
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """创建高斯窗口"""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """计算 SSIM"""
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        window = self.window
        
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """返回 1 - SSIM 作为损失（SSIM 越高损失越低）"""
        return 1.0 - self._ssim(img1, img2)


class L1SSIMLoss(nn.Module):
    """
    L1 Loss + SSIM Loss 组合
    
    L1 损失：确保像素级准确性
    SSIM 损失：确保结构相似性
    """
    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.l1_weight * l1 + self.ssim_weight * ssim


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss (感知损失)
    
    使用预训练的 VGG 提取特征，计算特征级别的差异
    这有助于保持生成图像的纹理和细节
    """
    def __init__(self, layers: list = None):
        super().__init__()
        self.layers = layers or ['relu1_2', 'relu2_2', 'relu3_3']
        
        # 使用 VGG16 作为特征提取器
        try:
            import torchvision.models as models
            vgg = models.vgg16(pretrained=True).features
            self.vgg = vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
            
            # 冻结 VGG 的 BN
            for m in self.vgg.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        except ImportError:
            self.vgg = None
            print("[Warning] torchvision not available, perceptual loss disabled")
        
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """预处理输入图像以匹配 VGG 期望的格式"""
        x = (x + 1) / 2  # 从 [-1, 1] 转换到 [0, 1]
        x = (x - self.mean) / self.std
        return x
    
    def _extract_features(self, x: torch.Tensor) -> list:
        """提取 VGG 特征"""
        if self.vgg is None:
            return []
        
        features = []
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in ['3', '8', '15']:  # relu1_2, relu2_2, relu3_3
                features.append(x)
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        
        pred_features = self._extract_features(self._preprocess(pred))
        target_features = self._extract_features(self._preprocess(target))
        
        loss = 0.0
        for pf, tf in zip(pred_features, target_features):
            loss += F.l1_loss(pf, tf)
        
        return loss / len(pred_features)


class GANLoss(nn.Module):
    """
    GAN 损失函数
    
    支持多种 GAN 损失类型：
    - vanilla: 标准交叉熵
    - lsgan: 最小二乘 GAN
    - wgan: Wasserstein GAN
    """
    def __init__(self, loss_type: str = 'lsgan'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'wgan':
            self.criterion = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, is_real: bool) -> torch.Tensor:
        if self.loss_type == 'vanilla':
            target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
            return self.criterion(pred, target)
        elif self.loss_type == 'lsgan':
            target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
            return self.criterion(pred, target)
        elif self.loss_type == 'wgan':
            return -pred.mean() if is_real else pred.mean()
        return torch.tensor(0.0)


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    包含：
    - L1 Loss: 像素级重建
    - SSIM Loss: 结构相似性
    - CSS Loss (可选): DAPI↔fake↔IHC 联合对比度-结构约束（CSSP2P GAN, 2025）
    - Edge Loss (可选)
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 1.0,
        edge_weight: float = 0.0,
        css_weight: float = 0.0,    # ← NEW：CSS Loss 权重，0 表示禁用
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.css_weight = css_weight
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()

        if css_weight > 0:
            self.css_loss = CSSLoss()
        else:
            self.css_loss = None
        
        # Sobel 边缘检测用于边缘损失
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self._init_sobel()
    
    def _init_sobel(self):
        """初始化 Sobel 算子"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_x.weight.data = sobel_x
        self.sobel_y.weight.data = sobel_y
    
    def _edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """边缘损失"""
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        pred_edge_x = self.sobel_x(pred_gray)
        pred_edge_y = self.sobel_y(pred_gray)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)
        target_edge_x = self.sobel_x(target_gray)
        target_edge_y = self.sobel_y(target_gray)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)
        return F.l1_loss(pred_edge, target_edge)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, dapi: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: 生成图 (-1, 1)
            target: 真实 IHC (-1, 1)
            dapi: 输入 DAPI/H&E (-1, 1)；css_weight > 0 时必传
        """
        loss = 0.0
        if self.l1_weight > 0:
            loss += self.l1_weight * self.l1_loss(pred, target)
        if self.ssim_weight > 0:
            loss += self.ssim_weight * self.ssim_loss(pred, target)
        # CSS Loss: DAPI↔fake↔IHC 三方约束 (CSSP2P GAN, 2025)
        if self.css_weight > 0 and self.css_loss is not None:
            assert dapi is not None, "css_weight > 0 时必须传入 dapi"
            loss += self.css_weight * self.css_loss(pred, dapi, target)
        if self.edge_weight > 0:
            loss += self.edge_weight * self._edge_loss(pred, target)
        return loss


# ---------------------------------------------------------------------------
# 新增：CSS Loss (Contrast-Structure Similarity)
# 论文：CSSP2P GAN (arXiv 2511.18946, 2025)
# 公式参考：PC-StainGAN (Liu et al., IEEE TMI 2021)
# ---------------------------------------------------------------------------

class CSSLoss(nn.Module):
    """
    Contrast-Structure Similarity (CSS) Loss。

    迫使生成图同时满足：
      1. 与输入 DAPI 在 Contrast + Structure 上相似（保留原始形态）
      2. 与目标 IHC 在 Contrast + Structure 上匹配（学习正确染色风格）

    公式：CSS(x,y) = -log( mean_{patch} ( (c(x,y)+s(x,y))/2 ) )
      c(x,y) = (2*mu_xy + eps) / (mu_x^2 + mu_y^2 + eps)   # 对比度项
      s(x,y) = (2*sigma_xy + eps) / (sigma_x^2 + sigma_y^2 + eps) # 结构项
    """
    def __init__(self, window_size: int = 11, channel: int = 3, eps: float = 1e-6):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.eps = eps
        self.register_buffer('window', self._create_window(window_size, channel))

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()
        _1D = gaussian(window_size, 1.5).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D.expand(channel, 1, window_size, window_size).contiguous()

    def _pair_stats(self, img1: torch.Tensor, img2: torch.Tensor):
        """返回 (mu1, mu2, mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12)。"""
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        w = self.window
        pad = self.window_size // 2
        mu1 = F.conv2d(img1, w, padding=pad, groups=self.channel)
        mu2 = F.conv2d(img2, w, padding=pad, groups=self.channel)
        mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, w, padding=pad, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, w, padding=pad, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, w, padding=pad, groups=self.channel) - mu1_mu2
        return mu1, mu2, mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12

    def _cs_map(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Contrast + Structure 组合 map，取 channel 平均。"""
        _, _, mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12 = self._pair_stats(img1, img2)
        c = (2 * mu1_mu2 + self.eps) / (mu1_sq + mu2_sq + self.eps)
        s = (2 * sigma12 + self.eps) / (sigma1_sq + sigma2_sq + self.eps)
        return ((c + s) / 2.0).mean(dim=1, keepdim=False)  # (B, H, W)

    def forward(self, fake: torch.Tensor, dapi: torch.Tensor,
                target: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            fake:   生成图 (B, 3, H, W)  [-1,1]
            dapi:   输入 DAPI     (B, 3, H, W)  [-1,1]
            target: 目标 IHC     (B, 3, H, W)  [-1,1]
        Returns:
            CSS Loss scalar（越小越好）
        """
        css_df = self._cs_map(dapi, fake)
        if target is None:
            # 用 soft-log 防止 log(0) 爆炸：log(x+eps) 的梯度在 x→0 时会趋近于 1/eps
            # 用 clamp 确保数值稳定
            return -torch.log(clamp_safe(css_df, eps=self.eps)).mean()

        css_ft = self._cs_map(fake, target)
        css_combined = 0.5 * (css_df + css_ft)
        return -torch.log(clamp_safe(css_combined, eps=self.eps)).mean()


# ---------------------------------------------------------------------------
# 新增：LPIPS (Learned Perceptual Image Patch Similarity)
# 论文：Zhang et al., "The Unreasonable Effectiveness of Deep Features as a
#       Perceptual Metric" (CVPR 2018)
# 优先用官方 lpips 库；不可用时回退到 VGG 多层 L1
# ---------------------------------------------------------------------------

class LPIPSLoss(nn.Module):
    """
    LPIPS 感知损失。与标准 PerceptualLoss 的区别：
      - 使用预训练网络 + 学习到的线性权重（与人类感知更对齐）
      - 在 histopathology 领域比纯 L1/VGG 更好（论文 HAPS, 2025）
    """
    def __init__(self, net: str = 'vgg'):
        super().__init__()
        self._impl = None
        self._fallback = None
        try:
            import lpips          # pip install lpips
            self._impl = lpips.LPIPS(net=net, verbose=False)
            for p in self._impl.parameters():
                p.requires_grad = False
            self._impl.eval()
        except Exception:
            # 回退：VGG16 多层 L1 平均（行为近似）
            try:
                from torchvision.models import vgg16, VGG16_Weights
                vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
                self._fallback = vgg.eval()
                for p in self._fallback.parameters():
                    p.requires_grad = False
                self.register_buffer('mean',
                    torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer('std',
                    torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            except Exception:
                self._fallback = None

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # [-1,1] → [0,1] → ImageNet normalize
        x = (x + 1) / 2.0
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std

    def _vgg_feats(self, x: torch.Tensor):
        feats = []
        for name, module in self._fallback._modules.items():
            x = module(x)
            if name in ('3', '8', '15'):   # relu1_2, relu2_2, relu3_3
                feats.append(x)
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   生成图 [-1,1]
            target: 真实图 [-1,1]
        Returns:
            LPIPS scalar（越小越好）
        """
        if self._impl is not None:
            return self._impl(pred, target).mean()
        if self._fallback is None:
            return (pred - target).abs().mean()
        pf = self._vgg_feats(self._preprocess(pred))
        tf = self._vgg_feats(self._preprocess(target))
        loss = sum(F.l1_loss(a, b) for a, b in zip(pf, tf))
        return loss / max(len(pf), 1)


# ---------------------------------------------------------------------------
# 新增：PyramidPix2Pix 多尺度损失
# 论文：Liu et al., "BCI: Breast Cancer Immunohistochemical Image Generation
#       through Pyramid Pix2pix" (CVPR 2022 Workshop)
# 核心思想：在 4 个尺度上同时用 L1 Loss 约束，让模型学习
#          "局部精确 + 全局一致" 的虚拟染色。
# ---------------------------------------------------------------------------

class GaussianPyramid(nn.Module):
    """
    高斯金字塔：对输入图像进行下采样 + 低通滤波

    Args:
        levels (int): 金字塔层数（论文用 4 层：L1=L1, L2=1/2, L3=1/4, L4=1/8）
        downscale (float): 每层下采样倍数（默认 2.0）

    Forward:
        x (B, C, H, W) → list of [x_L1, x_L2, ..., x_Ln]
    """
    def __init__(self, levels: int = 4, downscale: float = 2.0):
        super().__init__()
        self.levels = levels
        self.downscale = downscale

        # 高斯核（5×5, sigma=1）
        kernel_size = 5
        sigma = 1.0
        kernel = self._gaussian_kernel(kernel_size, sigma)
        # 扩展为分组卷积（每个通道一个核）
        self.register_buffer(
            'gaussian_kernel',
            kernel.unsqueeze(0).expand(3, -1, -1, -1).contiguous()
        )
        self.pad = kernel_size // 2

    @staticmethod
    def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        k = g.unsqueeze(1) @ g.unsqueeze(0)
        return k.unsqueeze(0)  # (1, 1, K, K)

    def _blur_down(self, x: torch.Tensor) -> torch.Tensor:
        """低通滤波 + 下采样"""
        # 边缘 padding (reflect) + grouped conv
        x = F.pad(x, [self.pad, self.pad, self.pad, self.pad], mode='reflect')
        x = F.conv2d(x, self.gaussian_kernel, groups=x.shape[1])
        # 下采样
        new_size = [int(s / self.downscale) for s in x.shape[2:]]
        return F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: 输入图像 (B, 3, H, W), 范围 [-1, 1]
        Returns:
            list of length `levels`:
                [x_L1=full, x_L2=1/2, x_L3=1/4, x_L4=1/8]
        """
        pyramid = [x]
        cur = x
        for _ in range(self.levels - 1):
            cur = self._blur_down(cur)
            pyramid.append(cur)
        return pyramid


class PyramidLoss(nn.Module):
    """
    PyramidPix2Pix 多尺度损失 (CVPR 2022 Workshop)。

    在 L1 + L2 + L3 + L4 四个尺度同时计算 L1 损失，迫使生成图：
      - 在 L1 (原始分辨率) 像素级精确
      - 在 L2 (1/2 分辨率) 局部结构一致
      - 在 L3 (1/4 分辨率) 中等结构一致
      - 在 L4 (1/8 分辨率) 全局布局一致

    论文 BCI 实验结论：
      pattern=L1_L2_L3_L4 (4 层)  >  L1_L2_L3 (3 层)  >  L1 (单层)

    Args:
        levels (int): 金字塔层数，默认 4
        weights (list of float): 每层损失权重，默认等权 [1, 1, 1, 1]
                                  可设 [1.0, 0.7, 0.5, 0.3] 强调低分辨率全局
    """
    def __init__(self, levels: int = 4, weights: list = None):
        super().__init__()
        self.levels = levels
        self.weights = weights or [1.0] * levels
        assert len(self.weights) == levels, \
            f"weights length {len(self.weights)} != levels {levels}"

        self.pyramid = GaussianPyramid(levels=levels, downscale=2.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   生成图 (B, 3, H, W) [-1, 1]
            target: 真实图 (B, 3, H, W) [-1, 1]
        Returns:
            多尺度 L1 加权求和（不归一化，梯度强度与单层 L1 一致）
        """
        pred_pyr = self.pyramid(pred)
        target_pyr = self.pyramid(target)

        loss = 0.0
        for w, p, t in zip(self.weights, pred_pyr, target_pyr):
            loss = loss + w * F.l1_loss(p, t)
        return loss


class PyramidL1SSIMLoss(nn.Module):
    """
    PyramidPix2Pix + SSIM 组合损失（v11 主力损失）。

    公式:
        L = λ_pyr * PyramidL1(pred, target)
          + λ_ssim * SSIM(pred, target)     ← 仅在原始分辨率

    相比 v6 的 PyramidL1SSIMLoss：
      - PyramidLoss 不再除以 sum(weights)，梯度强度与单层 L1 一致
      - SSIM 在原始分辨率（256x256）计算，保证细节锐利
      - 两项损失权重独立可调

    Args:
        pyramid_weight: 多尺度 L1 权重（默认 100，与原 L1 weight 对齐）
        ssim_weight: SSIM 权重（默认 50，与 v2 一致）
        levels: 金字塔层数（默认 4）
        pyr_weights: 每层金字塔权重（默认 [1, 1, 1, 1]）
    """
    def __init__(
        self,
        pyramid_weight: float = 100.0,
        ssim_weight: float = 50.0,
        levels: int = 4,
        pyr_weights: list = None,
    ):
        super().__init__()
        self.pyramid_weight = pyramid_weight
        self.ssim_weight = ssim_weight

        self.pyramid_loss = PyramidLoss(levels=levels, weights=pyr_weights)
        self.ssim_loss = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.pyramid_weight > 0:
            loss += self.pyramid_weight * self.pyramid_loss(pred, target)
        if self.ssim_weight > 0:
            loss += self.ssim_weight * self.ssim_loss(pred, target)
        return loss

