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
        # Persist the fixed validation decision inside the one final checkpoint.
        self.register_buffer('deployment_mask', torch.zeros(markers, dtype=torch.bool))
        self.register_buffer('deployment_locked', torch.tensor(False, dtype=torch.bool))

    def lock_deployment(self, mask):
        if self.deployment_locked.item():
            raise ValueError('Deployment is already locked')
        if len(mask) != self.deployment_mask.numel() or any(type(value) is not bool for value in mask):
            raise ValueError('Invalid deployment marker mask')
        self.deployment_mask.copy_(torch.tensor(mask, dtype=torch.bool, device=self.deployment_mask.device))
        self.deployment_locked.fill_(True)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(self, dapi):
        with torch.no_grad():
            base = self.baseline(dapi).float()
        if self.deployment_locked.item() and not self.deployment_mask.any().item():
            return base
        logits = self.refiner(torch.cat((dapi, base.to(dapi.dtype)), dim=1)).float()
        candidate = (base + self.residual_limit*logits.tanh()).clamp(0, 1)
        if not self.deployment_locked.item():
            return candidate
        mask = self.deployment_mask.view(1, -1, 1, 1)
        return torch.where(mask, candidate, base)
