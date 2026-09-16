"""ResNet50-UNet for DAPI->IHC virtual staining.

Architecture:
  1. Tile 1-ch DAPI to 3-ch (repeat) — uses pretrained ImageNet BN correctly.
  2. Standard pretrained ResNet50 encoder (ImageNet weights, fine-tuned).
  3. UNet-style decoder: bilinear upsampling from 8x8 to 256x256 at multiple scales,
     with skip connections. Simple F.interpolate avoids ConvTranspose2d size bugs.
  4. Output: sigmoid [0,1].

Decoder design (trainable):
  layer4: 2048ch 8x8
  proj:   2048 → 512
  up1:    512 → 256   (8x8 → 16x16)
  up2:    256+1024(skip3) → 512 → 256  (16x16 → 32x32)
  up3:    256+512(skip2) → 512 → 256  (32x32 → 64x64)
  up4:    256+256(skip1) → 256  (64x64 → 128x128)
  up5:    256 → 128  (128x128 → 256x256)
  head:   128 → markers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class DecoderBlock(nn.Module):
    """Reduce channels → bilinear upsample → concat skip → 2× conv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, skip_ch, 1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(skip_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.reduce(x)
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class ResNetUNet(nn.Module):
    """ResNet50 encoder + UNet decoder, sigmoid multi-marker output."""

    def __init__(self, markers: int = 4, pretrained: bool = True):
        super().__init__()
        backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        # Encoder (pretrained, fine-tuned)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1   # 256ch, 64x64
        self.layer2 = backbone.layer2   # 512ch, 32x32
        self.layer3 = backbone.layer3   # 1024ch, 16x16
        self.layer4 = backbone.layer4   # 2048ch, 8x8

        # Project layer4 to match decoder channel count
        self.proj = nn.Conv2d(2048, 512, 1)

        # Decoder: upsample from 8x8 to 256x256 with skip connections
        # Each block: upsample x2 + concat skip + conv
        self.up3 = DecoderBlock(512, 1024, 256)  # 16x16 ← layer3 skip
        self.up2 = DecoderBlock(256, 512, 256)   # 32x32 ← layer2 skip
        self.up1 = DecoderBlock(256, 256, 256)   # 64x64 ← layer1 skip

        # Final upsample: 64x64 → 256x256
        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Conv2d(128, markers, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, x):
        # Tile 1-ch → 3-ch (pretrained ImageNet BN expects 3 channels)
        x = x.repeat(1, 3, 1, 1)

        # Encoder
        x = self.relu(self.bn1(self.conv1(x)))    # /2, 64ch, 128x128
        x = self.maxpool(x)                       # /4, 64ch, 64x64
        s1 = self.layer1(x)                       # /4, 256ch, 64x64
        s2 = self.layer2(s1)                      # /8, 512ch, 32x32
        s3 = self.layer3(s2)                      # /16, 1024ch, 16x16
        s4 = self.layer4(s3)                      # /32, 2048ch, 8x8

        # Decoder
        p = self.proj(s4)                         # 512ch, 8x8
        d3 = self.up3(p, s3)                      # 256ch, 16x16
        d2 = self.up2(d3, s2)                     # 256ch, 32x32
        d1 = self.up1(d2, s1)                     # 256ch, 64x64
        out = self.up_final(d1)                   # 128ch, 256x256
        out = self.head(out)                       # markers, 256x256
        return torch.sigmoid(out)


def reconstruction_loss(pred, target):
    """Pixel L1 + local SSIM over [0,1] tensors."""
    l1 = (pred - target).abs().mean()
    C = pred.shape[1]
    ssim_vals = []
    for c in range(C):
        p, t = pred[:, c:c+1], target[:, c:c+1]
        mu_x = F.avg_pool2d(p, 5, 1, 2)
        mu_y = F.avg_pool2d(t, 5, 1, 2)
        sigma_x = F.avg_pool2d(p.pow(2), 5, 1, 2) - mu_x.pow(2)
        sigma_y = F.avg_pool2d(t.pow(2), 5, 1, 2) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(p * t, 5, 1, 2) - mu_x * mu_y
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2))
        ssim_vals.append(ssim.mean())
    ssim_loss = 1.0 - torch.stack(ssim_vals).mean()
    return l1 + 0.5 * ssim_loss
