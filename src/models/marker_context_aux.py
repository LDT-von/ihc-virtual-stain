"""MarkerContextNet with optional CD68-specific auxiliary head.

Architecture:
- Shared backbone (same as MarkerContextNet)
- 4 main heads (HLA-DR, CD68, CD45RO, Vimentin)
- 1 additional CD68-auxiliary head that takes the same features + a CD68-specific
  intermediate projection, encouraging the model to learn features dedicated to CD68.

This is a parallel branch, NOT a replacement. The CD68 main head is still trained,
and the auxiliary head provides additional gradient signal specifically for CD68.
"""
import torch
from torch import nn
from torch.nn import functional as F


# Import original components
from .marker_context import (ChannelNorm, GatedSpatialBlock, PooledContext,
                              local_ssim)


class CD68AuxNet(nn.Module):
    """MarkerContextNet + CD68-specific auxiliary branch."""
    CD68_INDEX = 1  # in (HLA-DR, CD68, CD45RO, Vimentin)

    def __init__(self, width=16, markers=4, context=True):
        super().__init__()
        if width < 4 or width % 4 or markers < 1:
            raise ValueError('width must be a positive multiple of four; markers must be positive')
        self.width = width
        self.markers = markers

        # === Shared backbone (same as MarkerContextNet) ===
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        channels = [width * (2 ** i) for i in range(4)]
        self.encoders = nn.ModuleList([nn.Sequential(*[GatedSpatialBlock(c) for _ in range(n)])
                                        for c, n in zip(channels, (1, 1, 2, 2))])
        self.down = nn.ModuleList([nn.Conv2d(c, c * 2, 2, stride=2) for c in channels[:-1]])
        self.context = nn.Sequential(GatedSpatialBlock(channels[-1], dilation=2),
                                     PooledContext(channels[-1])) if context else nn.Identity()
        self.up = nn.ModuleList([nn.Conv2d(c * 2, c, 1) for c in reversed(channels[:-1])])
        self.fuse = nn.ModuleList([nn.Conv2d(c * 2, c, 1) for c in reversed(channels[:-1])])
        self.decoders = nn.ModuleList([GatedSpatialBlock(c) for c in reversed(channels[:-1])])

        # === Main heads (one per marker) ===
        self.heads = nn.ModuleList([nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                                   nn.Conv2d(width, 1, 1)) for _ in range(markers)])
        for head in self.heads:
            nn.init.normal_(head[-1].weight, std=.01)
            nn.init.constant_(head[-1].bias, -2.0)

        # === CD68-auxiliary branch ===
        # Takes the final decoder feature (width channels) and runs an additional
        # GatedSpatialBlock + 3x3 conv + 1x1 conv to produce an additional CD68 prediction.
        self.cd68_aux_block = GatedSpatialBlock(width)
        self.cd68_aux_head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                            nn.Conv2d(width, 1, 1))
        nn.init.normal_(self.cd68_aux_head[-1].weight, std=.01)
        nn.init.constant_(self.cd68_aux_head[-1].bias, -2.0)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError('Expected N,1,H,W DAPI in [0,1]')
        h, w = x.shape[-2:]
        if min(h, w) < 8:
            raise ValueError('Image must be at least 8x8')
        x = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode='replicate')
        z, skips = self.stem(x), []
        for i, encoder in enumerate(self.encoders):
            z = encoder(z)
            if i < len(self.down):
                skips.append(z)
                z = self.down[i](z)
        z = self.context(z)
        for up, fuse, decoder, skip in zip(self.up, self.fuse, self.decoders, reversed(skips)):
            z = up(F.interpolate(z, size=skip.shape[-2:], mode='bilinear', align_corners=False))
            z = decoder(fuse(torch.cat((z, skip), dim=1)))
        # z is the final decoder feature (B, width, H', W')
        # === Main heads output ===
        main_outs = torch.cat([head(z) for head in self.heads], dim=1).sigmoid()[..., :h, :w]
        # === CD68-auxiliary output ===
        aux_feat = self.cd68_aux_block(z)
        cd68_aux_out = self.cd68_aux_head(aux_feat).sigmoid()[..., :h, :w]
        # Return (main, cd68_aux) for training; main alone for inference
        return main_outs, cd68_aux_out

    @torch.no_grad()
    def predict(self, x):
        """Inference: returns only main output."""
        main, _ = self.forward(x)
        return main


def reconstruction_loss_with_cd68_aux(pred_main, pred_aux, target, aux_weight=0.5):
    """Reconstruction loss + auxiliary CD68 supervision.

    pred_main: (B, 4, H, W) main heads output
    pred_aux: (B, 1, H, W) CD68-auxiliary head output (single channel)
    target: (B, 4, H, W) ground truth markers
    """
    p, t = pred_main.float(), target.float()
    # Main loss (all markers)
    main_loss = (1 - local_ssim(p, t).mean()
                 + 0.5 * F.l1_loss(p, t)
                 + 2 * F.mse_loss(p, t))
    for scale in (2, 4):
        main_loss = main_loss + 0.1 * F.l1_loss(F.avg_pool2d(p, scale),
                                                  F.avg_pool2d(t, scale))
    # CD68-aux loss (aux has 1 channel, target has CD68 at index 1)
    pa = pred_aux.float()
    ta = target[:, 1:2].float()  # CD68 channel
    aux_loss = (1 - local_ssim(pa, ta).mean()
                + 0.5 * F.l1_loss(pa, ta)
                + 2 * F.mse_loss(pa, ta))
    return main_loss + aux_weight * aux_loss
