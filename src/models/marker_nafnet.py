"""NAFNet-inspired DAPI-to-marker image translation, implemented with PyTorch.

The model predicts the selected marker channels directly from DAPI. It has
no image-to-output residual connection because DAPI and marker intensities
represent different modalities.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _ChannelLayerNorm(nn.Module):
    """Apply layer normalization across channels at each pixel."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch's layer_norm handles its reductions in a numerically stable way.
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias)
        return x.permute(0, 3, 1, 2)


class _SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class _NAFBlock(nn.Module):
    """Two gated residual branches with spatial mixing and channel attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = 2 * channels
        self.spatial_norm = _ChannelLayerNorm(channels)
        self.spatial_expand = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.spatial_gate = _SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
        )
        self.spatial_project = nn.Conv2d(channels, channels, 1)

        self.channel_norm = _ChannelLayerNorm(channels)
        self.channel_expand = nn.Conv2d(channels, hidden, 1)
        self.channel_gate = _SimpleGate()
        self.channel_project = nn.Conv2d(channels, channels, 1)

        # Zero-init residual scales make every block an identity at initialization.
        self.spatial_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.channel_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_expand(self.spatial_norm(x))
        spatial = self.spatial_gate(self.depthwise(spatial))
        spatial = self.spatial_project(spatial * self.channel_attention(spatial))
        x = x + self.spatial_scale * spatial

        channel = self.channel_expand(self.channel_norm(x))
        channel = self.channel_project(self.channel_gate(channel))
        return x + self.channel_scale * channel


class MarkerNAFNet(nn.Module):
    """Four-stage U-shaped network for one-channel DAPI to marker prediction.

    Args:
        markers: Number of independently predicted marker channels.
        width: Feature width of the full-resolution stage.

    Input must be a floating-point tensor of shape ``N x 1 x H x W`` with
    intensities in ``[0, 1]`` and ``H, W >= 16``. Output has shape
    ``N x markers x H x W`` and intensities in ``[0, 1]``. Spatial dimensions
    are internally padded to a multiple of 16 and then cropped back.
    """

    def __init__(self, markers: int = 4, width: int = 32) -> None:
        super().__init__()
        if isinstance(markers, bool) or not isinstance(markers, int) or markers < 1:
            raise ValueError("markers must be a positive integer")
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")

        self.markers = markers
        self.width = width
        levels = [width * (2**stage) for stage in range(4)]

        self.intro = nn.Conv2d(1, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for channels, depth in zip(levels, (1, 1, 2, 2)):
            self.encoders.append(nn.Sequential(*(_NAFBlock(channels) for _ in range(depth))))
            self.downsamples.append(nn.Conv2d(channels, 2 * channels, 2, stride=2))

        self.middle = nn.Sequential(_NAFBlock(16 * width), _NAFBlock(16 * width))
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for channels in reversed(levels):
            self.upsamples.append(
                nn.Sequential(
                    nn.Conv2d(2 * channels, 4 * channels, 1),
                    nn.PixelShuffle(2),
                )
            )
            self.decoders.append(_NAFBlock(channels))

        # Separate learnable filters for each stain, after the shared decoder.
        self.marker_heads = nn.ModuleList(
            nn.Sequential(
                _NAFBlock(width),
                nn.Conv2d(width, 1, 3, padding=1),
            )
            for _ in range(markers)
        )

    def forward(self, dapi: torch.Tensor) -> torch.Tensor:
        if not isinstance(dapi, torch.Tensor):
            raise TypeError("dapi must be a torch.Tensor")
        if dapi.ndim != 4 or dapi.shape[1] != 1:
            raise ValueError("dapi must have shape N x 1 x H x W")
        if not dapi.is_floating_point():
            raise TypeError("dapi must have a floating-point dtype")
        height, width = dapi.shape[-2:]
        if height < 16 or width < 16:
            raise ValueError("dapi height and width must each be at least 16")

        pad_h = (-height) % 16
        pad_w = (-width) % 16
        if pad_h or pad_w:
            dapi = F.pad(dapi, (0, pad_w, 0, pad_h), mode="reflect")

        features = self.intro(dapi)
        skips: list[torch.Tensor] = []
        for encoder, downsample in zip(self.encoders, self.downsamples):
            features = encoder(features)
            skips.append(features)
            features = downsample(features)

        features = self.middle(features)
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            features = decoder(upsample(features) + skip)

        marker_logits = torch.cat([head(features) for head in self.marker_heads], dim=1)
        return torch.sigmoid(marker_logits[..., :height, :width])


__all__ = ["MarkerNAFNet"]
