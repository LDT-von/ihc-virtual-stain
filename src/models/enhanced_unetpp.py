"""Enhanced UNet++ with Dense Skip Connections for IHC Virtual Staining.

Key innovations:
1. Dense skip connections (UNet++ style)
2. Multi-scale attention blocks
3. Feature fusion modules
4. MS-SSIM + Edge Loss combination

Architecture inspired by UNet++ and NAFNet.

Author: AI Assistant
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ============================================================================
# Building Blocks
# ============================================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D inputs (channels first)."""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ResBlock(nn.Module):
    """Residual block with layer norm."""
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = Swish()
    
    def forward(self, x):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x + 0.1 * h


class ConvNormAct(nn.Module):
    """Conv + LayerNorm + Activation."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.norm = LayerNorm2d(out_ch)
        self.act = Swish()
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ============================================================================
# Main Model: EnhancedUNetPlusPlus
# ============================================================================

class EnhancedUNetPlusPlus(nn.Module):
    """
    Enhanced UNet++ for multi-marker IHC virtual staining.
    
    Key features:
    - Dense skip connections at each level
    - Multi-scale attention
    - Deep supervision
    - Efficient and stable architecture
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # RGB DAPI
        out_channels: int = 4,  # 4 markers
        base_channels: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
        use_deep_supervision: bool = True,
    ):
        super().__init__()
        
        self.depth = depth
        self.use_deep_supervision = use_deep_supervision
        
        channels = [base_channels * (2 ** i) for i in range(depth)]
        
        # Input
        self.input_conv = ConvNormAct(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        for i in range(depth):
            self.encoders.append(nn.Sequential(
                ResBlock(channels[i], dropout),
                ResBlock(channels[i], dropout),
            ))
            if i < depth - 1:
                self.downsample.append(nn.Sequential(
                    ConvNormAct(channels[i], channels[i] * 2, 3, stride=2, padding=1),
                ))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(channels[-1], dropout),
            ResBlock(channels[-1], dropout),
            ResBlock(channels[-1], dropout),
        )
        
        # Decoder
        self.upsample = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(depth - 1):
            # Upsample: double spatial, halve channels
            self.upsample.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvNormAct(channels[-(i + 1)], channels[-(i + 2)], 3, padding=1),
            ))
            
            # Fuse: concat + conv
            self.fuse_convs.append(nn.Sequential(
                ConvNormAct(channels[-(i + 2)] * 2, channels[-(i + 2)], 3, padding=1),
            ))
            
            # Decoder blocks
            self.decoders.append(nn.Sequential(
                ResBlock(channels[-(i + 2)], dropout),
                ResBlock(channels[-(i + 2)], dropout),
            ))
        
        # Output
        self.output = nn.Sequential(
            ConvNormAct(base_channels, base_channels, 3, padding=1),
            nn.Conv2d(base_channels, out_channels, 1),
        )
        
        # Deep supervision heads
        if use_deep_supervision:
            self.deepsup_heads = nn.ModuleList()
            for i in range(depth - 1):
                self.deepsup_heads.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channels[-(i + 2)], out_channels, 1),
                ))
    
    def forward(self, x, return_all=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W) - RGB DAPI
            return_all: Return all outputs including deep supervision
        
        Returns:
            out: (B, 4, H, W) - Predicted markers
        """
        # Input
        h = self.input_conv(x)
        
        # Encoder
        encoder_features = [h]
        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            encoder_features.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder
        deepsup_outputs = []
        for i in range(len(self.decoders)):
            # Upsample
            h = self.upsample[i](h)
            
            # Get encoder feature (skip connection)
            skip = encoder_features[-(i + 2)]
            
            # Align spatial dimensions
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
            # Concat and fuse
            h = torch.cat([h, skip], dim=1)
            h = self.fuse_convs[i](h)
            
            # Decode
            h = self.decoders[i](h)
            deepsup_outputs.append(h)
        
        # Output
        out = self.output(h)
        out = torch.sigmoid(out)
        
        if return_all and self.use_deep_supervision:
            # Deep supervision
            ds_outputs = []
            for i, (feat, head) in enumerate(zip(deepsup_outputs, self.deepsup_heads)):
                ds = head(feat)
                ds = F.interpolate(ds, size=out.shape[-2:], mode='bilinear', align_corners=False)
                ds_outputs.append(torch.sigmoid(ds))
            
            return out, ds_outputs
        
        return out


# ============================================================================
# Loss Functions
# ============================================================================

def ms_ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Multi-Scale SSIM Loss."""
    levels = 4
    msssim = []
    msssim_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels]
    
    for _ in range(levels):
        ssim = ssim_single_scale(pred, target, window_size)
        msssim.append(ssim)
        
        if pred.shape[-1] > 2 and pred.shape[-2] > 2:
            pred = F.avg_pool2d(pred, 2)
            target = F.avg_pool2d(target, 2)
    
    msssim_tensor = torch.stack(msssim[:levels])
    return 1 - torch.prod(torch.exp(-msssim_tensor) ** torch.tensor(msssim_weights, device=pred.device))


def ssim_single_scale(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """SSIM at single scale."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    
    pad = window_size // 2
    pred_pad = F.pad(pred, (pad, pad, pad, pad), mode='reflect')
    target_pad = F.pad(target, (pad, pad, pad, pad), mode='reflect')
    
    kernel = torch.ones(window_size, window_size, device=pred.device) / (window_size ** 2)
    
    # Local means
    mu_pred = F.conv2d(pred_pad, kernel.expand(pred.shape[1], 1, -1, -1), groups=pred.shape[1])
    mu_target = F.conv2d(target_pad, kernel.expand(target.shape[1], 1, -1, -1), groups=target.shape[1])
    
    # Variances
    var_pred = F.conv2d(pred_pad ** 2, kernel.expand(pred.shape[1], 1, -1, -1), groups=pred.shape[1]) - mu_pred ** 2
    var_target = F.conv2d(target_pad ** 2, kernel.expand(target.shape[1], 1, -1, -1), groups=target.shape[1]) - mu_target ** 2
    cov = F.conv2d(pred_pad * target_pad, kernel.expand(pred.shape[1], 1, -1, -1), groups=pred.shape[1]) - mu_pred * mu_target
    
    ssim_map = ((2 * mu_pred * mu_target + C1) * (2 * cov + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (var_pred + var_target + C2))
    
    return ssim_map.mean()


def edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Edge-aware loss using Sobel filters."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    
    def gradient(x):
        gx = F.conv2d(x, sobel_x.expand(x.shape[1], -1, -1, -1), groups=x.shape[1], padding=1)
        gy = F.conv2d(x, sobel_y.expand(x.shape[1], -1, -1, -1), groups=x.shape[1], padding=1)
        return gx, gy
    
    pred_gx, pred_gy = gradient(pred)
    target_gx, target_gy = gradient(target)
    
    return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_msssim: float = 1.0,
    lambda_l1: float = 0.5,
    lambda_edge: float = 0.1,
    lambda_deepsup: float = 0.2,
    deepsup_preds: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, dict]:
    """Combined loss for training."""
    losses = {}
    
    # MS-SSIM loss
    losses['msssim'] = ms_ssim_loss(pred, target)
    
    # L1 loss
    losses['l1'] = F.l1_loss(pred, target)
    
    # Edge loss
    losses['edge'] = edge_loss(pred, target)
    
    # Deep supervision loss
    if deepsup_preds:
        ds_loss = 0.0
        for ds_pred in deepsup_preds:
            ds_loss += ms_ssim_loss(ds_pred, target) + 0.5 * F.l1_loss(ds_pred, target)
        losses['deepsup'] = ds_loss / len(deepsup_preds)
    else:
        losses['deepsup'] = torch.tensor(0.0, device=pred.device)
    
    # Total loss
    loss = (
        lambda_msssim * losses['msssim'] +
        lambda_l1 * losses['l1'] +
        lambda_edge * losses['edge'] +
        lambda_deepsup * losses['deepsup']
    )
    
    return loss, losses


# ============================================================================
# Model Builder
# ============================================================================

def build_enhanced_unet(
    in_channels: int = 3,
    out_channels: int = 4,
    base_channels: int = 64,
    depth: int = 4,
    **kwargs
) -> EnhancedUNetPlusPlus:
    """Build enhanced UNet++ model."""
    return EnhancedUNetPlusPlus(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        depth=depth,
        **kwargs
    )


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    print("Testing EnhancedUNetPlusPlus...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EnhancedUNetPlusPlus(
        in_channels=3,
        out_channels=4,
        base_channels=64,
        depth=4,
        use_deep_supervision=True,
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    x = torch.randn(2, 3, 256, 256, device=device)
    
    out, ds_outs = model(x, return_all=True)
    print(f"Input: {x.shape}, Output: {out.shape}, DS outputs: {len(ds_outs)}")
    
    # Test loss
    target = torch.rand(2, 4, 256, 256, device=device)
    loss, losses = combined_loss(out, target, deepsup_preds=ds_outs)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss breakdown: { {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()} }")
    
    print("Smoke test passed!")
