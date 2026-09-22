"""Deterministic joint DAPI-to-IHC regression with marker-specific decoders.

The encoder shares nuclear morphology. Every marker has a complete decoder,
so expression-specific spatial features need not share the final feature map.
This is an engineering candidate, not a claim of a new attention mechanism.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .marker_context import GatedSpatialBlock, PooledContext


class MarkerDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.ModuleList([nn.Conv2d(c*2, c, 1) for c in reversed(channels[:-1])])
        self.fuse = nn.ModuleList([nn.Conv2d(c*2, c, 1) for c in reversed(channels[:-1])])
        self.blocks = nn.ModuleList([nn.Sequential(GatedSpatialBlock(c), GatedSpatialBlock(c))
                                     for c in reversed(channels[:-1])])
        self.head = nn.Conv2d(channels[0], 1, 1)
        nn.init.normal_(self.head.weight, std=.01)
        nn.init.constant_(self.head.bias, -2.)

    def forward(self, z, skips):
        for up, fuse, block, skip in zip(self.up, self.fuse, self.blocks, reversed(skips)):
            z = up(F.interpolate(z, size=skip.shape[-2:], mode='bilinear', align_corners=False))
            z = block(fuse(torch.cat((z, skip), dim=1)))
        return self.head(z)


class MarkerSpecificNet(nn.Module):
    def __init__(self, width=24, markers=4, context=True, in_channels=1, logits=False):
        super().__init__()
        if width < 4 or width % 4 or markers < 1:
            raise ValueError('width must be a multiple of four; markers must be positive')
        channels = [width*2**i for i in range(4)]
        self.in_channels, self.logits = in_channels, logits
        self.stem = nn.Conv2d(in_channels, width, 3, padding=1)
        self.encoders = nn.ModuleList([nn.Sequential(*[GatedSpatialBlock(c) for _ in range(n)])
                                      for c, n in zip(channels, (2, 2, 3, 3))])
        self.down = nn.ModuleList([nn.Conv2d(c, c*2, 2, stride=2) for c in channels[:-1]])
        self.context = nn.Sequential(GatedSpatialBlock(channels[-1], dilation=2),
                                     PooledContext(channels[-1])) if context else nn.Identity()
        self.decoders = nn.ModuleList([MarkerDecoder(channels) for _ in range(markers)])

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels or min(x.shape[-2:]) < 8:
            raise ValueError(f'Expected N,{self.in_channels},H,W with H,W >= 8')
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode='replicate')
        z, skips = self.stem(x), []
        for i, encoder in enumerate(self.encoders):
            z = encoder(z)
            if i < len(self.down):
                skips.append(z)
                z = self.down[i](z)
        z = self.context(z)
        output = torch.cat([decoder(z, skips) for decoder in self.decoders], dim=1)[..., :h, :w]
        return output if self.logits else output.sigmoid()
