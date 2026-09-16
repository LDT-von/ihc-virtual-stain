"""MarkerContextNet: paired pixel regression, not a diffusion-model reproduction.

Shared multiscale morphology, pooled context attention and separate marker heads.
No direct DAPI-to-output residual: marker expression need not equal nuclear signal.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        variance, mean = torch.var_mean(x, dim=1, keepdim=True, correction=0)
        return ((x-mean) * torch.rsqrt(variance+1e-6) * self.weight + self.bias).to(dtype)


class GatedSpatialBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.expand = nn.Conv2d(channels, channels*2, 1)
        self.spatial = nn.Conv2d(channels*2, channels*2, 5, padding=2*dilation,
                                 dilation=dilation, groups=channels*2)
        self.channel_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x):
        a, b = self.spatial(self.expand(self.norm(x))).chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        return x + self.scale * self.project(y * self.channel_gate(y))


class PooledContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        h, w = min(8, x.shape[-2]), min(8, x.shape[-1])
        tokens = F.adaptive_avg_pool2d(x, (h, w)).flatten(2).transpose(1, 2)
        q = self.norm(tokens)
        context = self.attention(q, q, q, need_weights=False)[0]
        context = context.transpose(1, 2).reshape(x.shape[0], x.shape[1], h, w)
        return x + self.scale * F.interpolate(self.project(context), x.shape[-2:], mode='bilinear', align_corners=False)


class MarkerContextNet(nn.Module):
    def __init__(self, width=16, markers=4, context=True):
        super().__init__()
        if width < 4 or width % 4 or markers < 1:
            raise ValueError('width must be a positive multiple of four; markers must be positive')
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        channels = [width * (2**i) for i in range(4)]
        self.encoders = nn.ModuleList([nn.Sequential(*[GatedSpatialBlock(c) for _ in range(n)])
                                      for c, n in zip(channels, (1, 1, 2, 2))])
        self.down = nn.ModuleList([nn.Conv2d(c, c*2, 2, stride=2) for c in channels[:-1]])
        self.context = nn.Sequential(GatedSpatialBlock(channels[-1], dilation=2),
                                     PooledContext(channels[-1])) if context else nn.Identity()
        self.up = nn.ModuleList([nn.Conv2d(c*2, c, 1) for c in reversed(channels[:-1])])
        self.fuse = nn.ModuleList([nn.Conv2d(c*2, c, 1) for c in reversed(channels[:-1])])
        self.decoders = nn.ModuleList([GatedSpatialBlock(c) for c in reversed(channels[:-1])])
        self.heads = nn.ModuleList([nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                                 nn.Conv2d(width, 1, 1)) for _ in range(markers)])
        for head in self.heads:
            nn.init.normal_(head[-1].weight, std=.01)
            nn.init.constant_(head[-1].bias, -2.0)

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
        return torch.cat([head(z) for head in self.heads], dim=1).sigmoid()[..., :h, :w]


def local_ssim(pred, target):
    """N,C SSIM; uniform valid 7x7, sample covariance, [0,1].

    Matches skimage's default (non-Gaussian) SSIM, including its cropped border.
    FP32 statistics are deliberate even under autocast.
    """
    if pred.shape != target.shape or pred.ndim != 4 or min(pred.shape[-2:]) < 7:
        raise ValueError('SSIM requires equal NCHW tensors, spatial dimensions >=7')
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, t = pred.float(), target.float()
        stats = F.avg_pool2d(torch.cat((p, t, p*p, t*t, p*t), dim=1), 7, stride=1)
        mp, mt, pp, tt, pt = stats.chunk(5, dim=1)
        vp, vt, cov = (pp-mp*mp)*49/48, (tt-mt*mt)*49/48, (pt-mp*mt)*49/48
        score = ((2*mp*mt+.01**2)*(2*cov+.03**2)/
                 ((mp*mp+mt*mt+.01**2)*(vp+vt+.03**2)))
        return score.mean(dim=(-2, -1))


def reconstruction_loss(pred, target, marker_weights=None):
    """Adaptive per-marker pixel regression loss.

    `marker_weights` (C,) rebalances L1/MSE/multi-scale terms across the C marker
    channels so difficult markers (e.g. sparse Vimentin) get extra gradient. SSIM
    is left untouched because it is a structural term shared across channels.
    """
    p, t = pred.float(), target.float()
    if marker_weights is None:
        marker_weights = torch.ones(pred.shape[1], device=pred.device)
    marker_weights = marker_weights / marker_weights.sum().clamp_min(1e-6) * pred.shape[1]
    diff = p - t
    l1 = (diff.abs() * marker_weights.view(1, -1, 1, 1)).mean()
    mse = (diff.pow(2) * marker_weights.view(1, -1, 1, 1)).mean()
    loss = 1 - local_ssim(p, t).mean() + .5 * l1 + 2 * mse
    # Supervision uses true marker targets only, never DAPI as a surrogate label.
    for scale in (2, 4):
        loss = loss + .1 * (F.avg_pool2d(diff.abs(), scale)
                            * marker_weights.view(1, -1, 1, 1)).mean()
    return loss
