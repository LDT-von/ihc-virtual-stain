"""DAPI-only multi-marker regression with shared and marker-specific prototypes.

This is an original, lightweight architecture inspired by the *idea* of
shared/task-specific prototypes in ProtoMTG. It does not reproduce that model
or use its source code or weights.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .marker_context import GatedSpatialBlock, PooledContext
from .marker_specific import MarkerDecoder


class PrototypeRouter(nn.Module):
    """Route bottleneck features through shared and per-marker prototype banks."""

    def __init__(self, channels, markers, num_shared_prototypes, num_task_prototypes,
                 temperature):
        super().__init__()
        self.markers = markers
        self.temperature = temperature
        self.shared = nn.Parameter(torch.empty(num_shared_prototypes, channels))
        self.private = nn.Parameter(torch.empty(markers, num_task_prototypes, channels))
        nn.init.normal_(self.shared, std=channels ** -0.5)
        nn.init.normal_(self.private, std=channels ** -0.5)
        self.project = nn.ModuleList([nn.Conv2d(channels * 2, channels, 1)
                                      for _ in range(markers)])
        self.scale = nn.Parameter(torch.full((markers,), 0.1))

    def _bank(self, marker_index):
        return torch.cat((self.shared, self.private[marker_index]), dim=0)

    def forward(self, features, marker_index):
        if marker_index not in range(self.markers):
            raise IndexError(f'Invalid marker index: {marker_index}')
        # Keep cosine similarity and softmax in FP32 under autocast.
        with torch.autocast(device_type=features.device.type, enabled=False):
            normalized_features = F.normalize(features.float(), dim=1)
            bank = F.normalize(self._bank(marker_index).float(), dim=1)
            similarity = torch.einsum('bchw,kc->bkhw', normalized_features, bank)
            attention = torch.softmax(similarity / self.temperature, dim=1)
            context = torch.einsum('bkhw,kc->bchw', attention, bank)
        context = context.to(features.dtype)
        correction = self.project[marker_index](torch.cat((features, context), dim=1))
        return features + self.scale[marker_index] * correction

    def diversity_loss(self):
        """Penalize duplicate directions within each marker's available bank."""
        penalties = []
        for marker_index in range(self.markers):
            bank = F.normalize(self._bank(marker_index).float(), dim=1)
            gram = bank @ bank.T
            off_diagonal = gram - torch.diag_embed(gram.diagonal())
            count = bank.shape[0] * (bank.shape[0] - 1)
            penalties.append(off_diagonal.square().sum() / count)
        return torch.stack(penalties).mean()


class PrototypeMarkerNet(nn.Module):
    """Shared morphology encoder, prototype routing, four independent decoders."""

    def __init__(self, width=32, markers=4, context=True, num_shared_prototypes=8,
                 num_task_prototypes=4, temperature=0.25):
        super().__init__()
        if width < 4 or width % 4 or markers < 1:
            raise ValueError('width must be a positive multiple of four; markers must be positive')
        if num_shared_prototypes < 1 or num_task_prototypes < 1:
            raise ValueError('Both prototype counts must be positive')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Prototype temperature must be finite and positive')

        channels = [width * (2 ** index) for index in range(4)]
        self.stem = nn.Conv2d(1, width, 3, padding=1)
        self.encoders = nn.ModuleList([
            nn.Sequential(*[GatedSpatialBlock(channel) for _ in range(depth)])
            for channel, depth in zip(channels, (1, 1, 2, 2))
        ])
        self.down = nn.ModuleList([nn.Conv2d(channel, channel * 2, 2, stride=2)
                                   for channel in channels[:-1]])
        self.context = (nn.Sequential(GatedSpatialBlock(channels[-1], dilation=2),
                                      PooledContext(channels[-1]))
                        if context else nn.Identity())
        self.router = PrototypeRouter(channels[-1], markers, num_shared_prototypes,
                                      num_task_prototypes, temperature)
        self.decoders = nn.ModuleList([MarkerDecoder(channels) for _ in range(markers)])

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or min(x.shape[-2:]) < 8:
            raise ValueError('Expected N,1,H,W DAPI with H,W >= 8')
        height, width = x.shape[-2:]
        x = F.pad(x, (0, (-width) % 8, 0, (-height) % 8), mode='replicate')
        features, skips = self.stem(x), []
        for index, encoder in enumerate(self.encoders):
            features = encoder(features)
            if index < len(self.down):
                skips.append(features)
                features = self.down[index](features)
        features = self.context(features)
        outputs = [decoder(self.router(features, marker_index), skips)
                   for marker_index, decoder in enumerate(self.decoders)]
        return torch.cat(outputs, dim=1).sigmoid()[..., :height, :width]

    def prototype_diversity_loss(self):
        return self.router.diversity_loss()
