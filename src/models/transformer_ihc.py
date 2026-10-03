"""Efficient Models for IHC Virtual Staining.

Two architectures:
1. AttentionUNet: UNet with SE attention (recommended for speed)
2. HybridCNFTransformer: CNN + Attention hybrid

Author: AI Assistant
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ============================================================================
# Building Blocks
# ============================================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D inputs."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x_norm = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x_norm + self.bias[:, None, None]


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)
    
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = Swish()(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResBlock(nn.Module):
    """Residual block with SE attention."""
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = LayerNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = LayerNorm2d(channels)
        self.se = SEBlock(channels)
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
        
        h = self.se(h)
        return x + 0.1 * h


# ============================================================================
# Attention UNet
# ============================================================================

class AttentionUNet(nn.Module):
    """
    UNet with SE attention blocks.
    
    Architecture:
    - Encoder: ResBlocks + Downsample
    - Bottleneck: ResBlocks + SE
    - Decoder: Upsample + Concat + Fuse + ResBlocks
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        base_channels: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Channel list
        channels = [base_channels * (2 ** i) for i in range(depth)]
        
        # Input
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            LayerNorm2d(base_channels),
            Swish(),
        )
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, ch in enumerate(channels):
            self.encoders.append(nn.Sequential(
                ResBlock(ch, dropout),
                ResBlock(ch, dropout),
            ))
            if i < depth - 1:
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(ch, channels[i + 1], 3, stride=2, padding=1),
                    LayerNorm2d(channels[i + 1]),
                    Swish(),
                ))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(channels[-1], dropout),
            ResBlock(channels[-1], dropout),
            SEBlock(channels[-1]),
        )
        
        # Decoder
        self.upsamples = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(depth - 1):
            # Upsample
            self.upsamples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(channels[-(i + 1)], channels[-(i + 2)], 3, padding=1),
                LayerNorm2d(channels[-(i + 2)]),
                Swish(),
            ))
            
            # Fuse concat channels
            self.fuse_convs.append(nn.Sequential(
                nn.Conv2d(channels[-(i + 2)] * 2, channels[-(i + 2)], 1),
                LayerNorm2d(channels[-(i + 2)]),
                Swish(),
            ))
            
            # Decoder blocks
            self.decoders.append(nn.Sequential(
                ResBlock(channels[-(i + 2)], dropout),
                SEBlock(channels[-(i + 2)]),
            ))
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            LayerNorm2d(base_channels),
            Swish(),
            nn.Conv2d(base_channels, out_channels, 1),
        )
    
    def forward(self, x):
        # Input
        h = self.input_conv(x)
        
        # Encoder
        encoder_features = []
        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            encoder_features.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder
        for i in range(len(self.decoders)):
            # Upsample
            h = self.upsamples[i](h)
            
            # Skip connection
            skip = encoder_features[-(i + 2)]
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
            # Concat and fuse
            h = torch.cat([h, skip], dim=1)
            h = self.fuse_convs[i](h)
            
            # Decode
            h = self.decoders[i](h)
        
        # Output
        out = self.output(h)
        return torch.sigmoid(out)


# ============================================================================
# Hybrid CNN-Attention Model
# ============================================================================

class HybridCNFTransformer(nn.Module):
    """
    Hybrid model with multi-scale processing.
    
    Features:
    - Multi-scale feature extraction
    - Attention-based feature refinement
    - Residual connections
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        base_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Input
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            LayerNorm2d(base_channels),
            Swish(),
        )
        
        # Stage 1: base resolution
        self.stage1 = nn.Sequential(
            ResBlock(base_channels, dropout),
            ResBlock(base_channels, dropout),
            SEBlock(base_channels),
        )
        
        # Stage 2: downsample x2
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            LayerNorm2d(base_channels * 2),
            Swish(),
        )
        
        self.stage2 = nn.Sequential(
            ResBlock(base_channels * 2, dropout),
            ResBlock(base_channels * 2, dropout),
            SEBlock(base_channels * 2),
        )
        
        # Stage 3: downsample x2
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            LayerNorm2d(base_channels * 4),
            Swish(),
        )
        
        self.stage3 = nn.Sequential(
            ResBlock(base_channels * 4, dropout),
            ResBlock(base_channels * 4, dropout),
            ResBlock(base_channels * 4, dropout),
            SEBlock(base_channels * 4),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(base_channels * 4, dropout),
            ResBlock(base_channels * 4, dropout),
            SEBlock(base_channels * 4),
        )
        
        # Up 3
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            LayerNorm2d(base_channels * 2),
            Swish(),
        )
        
        self.fuse3 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 1),
            LayerNorm2d(base_channels * 2),
            Swish(),
        )
        
        self.stage3_up = nn.Sequential(
            ResBlock(base_channels * 2, dropout),
            SEBlock(base_channels * 2),
        )
        
        # Up 2
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            LayerNorm2d(base_channels),
            Swish(),
        )
        
        self.fuse2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 1),
            LayerNorm2d(base_channels),
            Swish(),
        )
        
        self.stage2_up = nn.Sequential(
            ResBlock(base_channels, dropout),
            ResBlock(base_channels, dropout),
            SEBlock(base_channels),
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            LayerNorm2d(base_channels),
            Swish(),
            nn.Conv2d(base_channels, out_channels, 1),
        )
    
    def forward(self, x):
        # Input
        h = self.input_conv(x)
        
        # Stage 1
        s1 = self.stage1(h)
        
        # Stage 2
        h = self.down2(s1)
        s2 = self.stage2(h)
        
        # Stage 3
        h = self.down3(s2)
        h = self.stage3(h)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Up 3
        h = self.up3(h)
        if h.shape != s2.shape:
            h = F.interpolate(h, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        h = self.fuse3(torch.cat([h, s2], dim=1))
        h = self.stage3_up(h)
        
        # Up 2
        h = self.up2(h)
        if h.shape != s1.shape:
            h = F.interpolate(h, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        h = self.fuse2(torch.cat([h, s1], dim=1))
        h = self.stage2_up(h)
        
        # Output
        out = self.output(h)
        return torch.sigmoid(out)


# ============================================================================
# Model Builders
# ============================================================================

def build_attention_unet(
    in_channels: int = 3,
    out_channels: int = 4,
    base_channels: int = 64,
    depth: int = 4,
    **kwargs
) -> AttentionUNet:
    """Build Attention UNet model."""
    return AttentionUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        depth=depth,
        **kwargs
    )


def build_hybrid_cnf_ihc(
    in_channels: int = 3,
    out_channels: int = 4,
    base_channels: int = 64,
    **kwargs
) -> HybridCNFTransformer:
    """Build hybrid CNN-Attention model."""
    return HybridCNFTransformer(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        **kwargs
    )


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    print("Testing models...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Attention UNet
    print("\n1. Attention UNet:")
    model = AttentionUNet(
        in_channels=3,
        out_channels=4,
        base_channels=32,
        depth=3,
    ).to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(2, 3, 256, 256, device=device)
    out = model(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    
    # Test Hybrid
    print("\n2. Hybrid CNN-Attention:")
    model2 = HybridCNFTransformer(
        in_channels=3,
        out_channels=4,
        base_channels=32,
    ).to(device)
    print(f"   Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    out2 = model2(x)
    print(f"   Input: {x.shape}, Output: {out2.shape}")
    
    print("\nAll tests passed!")
