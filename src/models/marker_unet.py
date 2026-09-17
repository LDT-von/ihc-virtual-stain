"""Per-marker U-Net for DAPI → single IHC marker virtual staining.

Key design choices:
1. Single-marker output (one model per marker, no shared encoder)
2. Input: DAPI 1-channel [0,1] → output: IHC 1-channel [0,1]
3. Heavy skip connections: full resolution (256x256) information preserved
4. GroupNorm + SiLU instead of BatchNorm (small batch, 1-channel input)
5. Optional DAPI residual: output = DAPI + tanh(decoder_output) * alpha
   - Allows model to either keep DAPI structure or learn marker from scratch
6. Attention gates in skip paths for selective feature passing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    """GroupNorm that works with any channel count."""
    def __init__(self, channels, groups=None, eps=1e-6):
        super().__init__()
        if groups is None:
            groups = max(1, channels // 4)
        # Clamp groups so it divides channels
        groups = max(1, min(groups, channels))
        self.gn = nn.GroupNorm(groups, channels, eps=eps)
    
    def forward(self, x):
        return self.gn(x)


class AttentionGate(nn.Module):
    """Gate signal from decoder to skip connection for selective feature passing.
    
    Both inputs are at the SAME spatial resolution but may have different channels.
    """
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_x = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, gate, skip):
        # gate: (B, gate_ch, H, W), skip: (B, skip_ch, H, W), same H,W
        a1 = self.W_g(gate)          # → (B, inter_ch, H, W)
        a2 = self.W_x(skip)          # → (B, inter_ch, H, W)
        psi = self.relu(a1 + a2)
        psi = self.sigmoid(self.psi(psi))
        return skip * psi


class ConvBlock(nn.Module):
    """Double conv: GN → SiLU → Conv → GN → SiLU → Conv."""
    def __init__(self, in_ch, out_ch, groups=None):
        super().__init__()
        self.block = nn.Sequential(
            GroupNorm(in_ch, groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            GroupNorm(out_ch, groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        return self.block(x) + self.residual(x)


class DownBlock(nn.Module):
    """Downsample with ConvBlock."""
    def __init__(self, in_ch, out_ch, groups=None):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, groups)
    
    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample + concat with attention gate + ConvBlock.

    Args:
        in_ch: channels from decoder (before upsample)
        skip_ch: channels from encoder skip connection
        out_ch: output channels after concat+conv
        attention: whether to use attention gate
    """
    def __init__(self, in_ch, skip_ch, out_ch, attention=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        inter_ch = min(in_ch, skip_ch) // 2
        if attention and inter_ch > 0:
            self.attn = AttentionGate(in_ch, skip_ch, inter_ch)
        else:
            self.attn = None
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)  # (B, in_ch, H*2, W*2)
        # Ensure spatial match before attention (skip may be from a different level)
        if self.attn is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            skip = self.attn(x, skip)  # both at same H,W
        else:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MarkerUNet(nn.Module):
    """
    Lightweight U-Net for single marker prediction.
    
    Architecture:
    - Input: 1×256×256 DAPI
    - Encoder: 64→128→256→512 with MaxPool
    - Bottleneck: 512 with self-attention
    - Decoder: 512→256→128→64 with attention gates
    - Output head: single channel sigmoid
    - Optional DAPI residual: out = sigmoid(out + DAPI * residual_weight)
    """
    
    def __init__(self, base_ch=64, residual_weight=0.0, use_attention=True):
        super().__init__()
        self.residual_weight = residual_weight
        
        # Stem: 1ch → base_ch
        groups_stem = max(1, base_ch // 4)
        self.stem = ConvBlock(1, base_ch, groups_stem)
        
        # Encoder
        self.enc1 = DownBlock(base_ch, base_ch * 2)       # 128x128
        self.enc2 = DownBlock(base_ch * 2, base_ch * 4)   # 64x64
        self.enc3 = DownBlock(base_ch * 4, base_ch * 8)   # 32x32
        
        # Bottleneck: no pooling, just conv
        groups_bottleneck = max(1, base_ch * 8 // 4)
        self.bottleneck = nn.Sequential(
            GroupNorm(base_ch * 8, groups_bottleneck),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch * 8, base_ch * 8, 3, padding=1, bias=False),
            GroupNorm(base_ch * 8, groups_bottleneck),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch * 8, base_ch * 8, 3, padding=1, bias=False),
        )
        
        # Self-attention at bottleneck
        nh = max(1, base_ch * 8 // 64)
        self.self_attn = nn.MultiheadAttention(base_ch * 8, nh, batch_first=True, dropout=0.0)
        self.attn_norm = GroupNorm(base_ch * 8, groups_bottleneck)
        
        # Decoder
        self.up3 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4, attention=use_attention)  # 64x64
        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, attention=use_attention)  # 128x128
        self.up1 = UpBlock(base_ch * 2, base_ch * 2, base_ch, attention=use_attention)       # 256x256
        
        # Output head: full-res fusion with skip from stem
        if use_attention:
            self.out_attn = AttentionGate(base_ch, base_ch, base_ch // 2)
        else:
            self.out_attn = None
        
        groups_head = max(1, base_ch // 4)
        self.head = nn.Sequential(
            GroupNorm(base_ch * 2, groups_head),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1, bias=False),
            GroupNorm(base_ch, groups_head),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 1, 1),
        )
        
        # Initialize output bias for slight positivity
        nn.init.constant_(self.head[-1].bias, -2.0)
    
    def _self_attention(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        x = x + self.attn_norm(attn_out.transpose(1, 2).reshape(B, C, H, W))
        return x
    
    def forward(self, dapi):
        if dapi.ndim != 4 or dapi.shape[1] != 1:
            raise ValueError('Expected DAPI: N,1,H,W in [0,1]')
        
        # Encoder
        x0 = self.stem(dapi)                              # B,64,256,256
        x1 = self.enc1(x0)                                # B,128,128,128
        x2 = self.enc2(x1)                                # B,256,64,64
        x3 = self.enc3(x2)                                # B,512,32,32
        
        # Bottleneck
        x = self.bottleneck(x3)                           # B,512,32,32
        x = self._self_attention(x)                       # B,512,32,32
        
        # Decoder
        x = self.up3(x, x3)                               # B,256,64,64
        x = self.up2(x, x2)                               # B,128,128,128
        x = self.up1(x, x1)                               # B,64,256,256
        
        # Full-res fusion
        if self.out_attn is not None:
            x0 = self.out_attn(x, x0)
        x = torch.cat([x, x0], dim=1)
        
        out = self.head(x).sigmoid()
        
        # Optional DAPI residual
        if self.residual_weight > 0:
            out = dapi * self.residual_weight + out * (1 - self.residual_weight)
        
        return out


class DeeperMarkerUNet(nn.Module):
    """
    Deeper variant with more capacity.
    Encoder: 1→64→128→256→512→1024 (5 levels, 32×32 bottleneck)
    Uses dropout for regularization.
    """
    
    def __init__(self, base_ch=48, dropout=0.1, residual_weight=0.0, use_attention=True):
        super().__init__()
        self.residual_weight = residual_weight
        
        groups_s = max(1, base_ch // 4)
        self.stem = ConvBlock(1, base_ch, groups_s)
        
        self.enc1 = DownBlock(base_ch, base_ch * 2)       # 128x128
        self.enc2 = DownBlock(base_ch * 2, base_ch * 4)   # 64x64
        self.enc3 = DownBlock(base_ch * 4, base_ch * 8)   # 32x32
        self.enc4 = DownBlock(base_ch * 8, base_ch * 16)  # 16x16
        
        # Bottleneck
        gb = max(1, base_ch * 16 // 8)
        self.bottleneck = nn.Sequential(
            GroupNorm(base_ch * 16, gb),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_ch * 16, base_ch * 16, 3, padding=1, bias=False),
            GroupNorm(base_ch * 16, gb),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_ch * 16, base_ch * 16, 3, padding=1, bias=False),
        )
        
        # Attention at bottleneck
        nh = max(1, base_ch * 16 // 64)
        self.self_attn = nn.MultiheadAttention(base_ch * 16, nh, batch_first=True, dropout=0.0)
        self.attn_norm = GroupNorm(base_ch * 16, gb)
        
        # Decoder
        self.up4 = UpBlock(base_ch * 16, base_ch * 16, base_ch * 8, attention=use_attention)  # 32x32
        self.up3 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4, attention=use_attention)     # 64x64
        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, attention=use_attention)    # 128x128
        self.up1 = UpBlock(base_ch * 2, base_ch * 2, base_ch, attention=use_attention)          # 256x256
        
        if use_attention:
            self.out_attn = AttentionGate(base_ch, base_ch, base_ch // 2)
        else:
            self.out_attn = None
        
        gh = max(1, base_ch // 4)
        self.head = nn.Sequential(
            GroupNorm(base_ch * 2, gh),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1, bias=False),
            GroupNorm(base_ch, gh),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 1, 1),
        )
        nn.init.constant_(self.head[-1].bias, -2.0)
    
    def _self_attention(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        x = x + self.attn_norm(attn_out.transpose(1, 2).reshape(B, C, H, W))
        return x
    
    def forward(self, dapi):
        x0 = self.stem(dapi)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        
        x = self.bottleneck(x4)
        x = self._self_attention(x)
        
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        
        if self.out_attn is not None:
            x0 = self.out_attn(x, x0)
        x = torch.cat([x, x0], dim=1)
        
        out = self.head(x).sigmoid()
        
        if self.residual_weight > 0:
            out = dapi * self.residual_weight + out * (1 - self.residual_weight)
        
        return out
