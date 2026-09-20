"""Frozen paired-regression baseline plus bounded cross-marker residual refinement."""
import torch
from torch import nn

from .marker_context import MarkerContextNet
from .marker_specific import MarkerSpecificNet


class AnchoredIHC(nn.Module):
    def __init__(self, baseline_config, width=24, markers=4, context=True, residual_limit=.2):
        super().__init__()
        config = dict(baseline_config)
        architecture = config.pop('architecture', 'context')
        if architecture not in ('context', 'marker_specific'):
            raise ValueError('Baseline must be a verified MarkerContextNet or MarkerSpecificNet checkpoint')
        if config.get('markers', 4) != markers or not 0 < residual_limit <= 1:
            raise ValueError('Invalid marker count or residual limit')
        self.baseline = (MarkerContextNet if architecture == 'context' else MarkerSpecificNet)(**config)
        self.baseline.requires_grad_(False).eval()
        self.refiner = MarkerSpecificNet(width, markers, context, in_channels=1+markers, logits=True)
        for decoder in self.refiner.decoders:
            nn.init.zeros_(decoder.head.weight)
            nn.init.zeros_(decoder.head.bias)
        self.residual_limit = float(residual_limit)
        # Runtime model-selection decision; training always optimizes all branches.
        self.deployment_mask = None

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(self, dapi):
        with torch.no_grad():
            base = self.baseline(dapi).float()
        if self.deployment_mask is not None and not any(self.deployment_mask):
            return base
        logits = self.refiner(torch.cat((dapi, base.to(dapi.dtype)), dim=1)).float()
        candidate = (base + self.residual_limit*logits.tanh()).clamp(0, 1)
        if self.deployment_mask is None:
            return candidate
        mask = torch.tensor(self.deployment_mask, dtype=torch.bool, device=dapi.device).view(1, -1, 1, 1)
        return torch.where(mask, candidate, base)
