"""SSIM Loss + L1 Loss + Perceptual Loss 用于 Pix2Pix GAN 训练"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 1.0,
        edge_weight: float = 0.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        
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
        # 转换为灰度图
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        
        # Sobel 边缘检测
        pred_edge_x = self.sobel_x(pred_gray)
        pred_edge_y = self.sobel_y(pred_gray)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)
        
        target_edge_x = self.sobel_x(target_gray)
        target_edge_y = self.sobel_y(target_gray)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)
        
        return F.l1_loss(pred_edge, target_edge)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        
        # L1 Loss
        if self.l1_weight > 0:
            loss += self.l1_weight * self.l1_loss(pred, target)
        
        # SSIM Loss
        if self.ssim_weight > 0:
            loss += self.ssim_weight * self.ssim_loss(pred, target)
        
        # Edge Loss
        if self.edge_weight > 0:
            loss += self.edge_weight * self._edge_loss(pred, target)
        
        return loss
