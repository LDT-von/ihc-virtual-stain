"""Pinned SMP ResNet34 U-Net adapted to continuous DAPI-to-marker regression."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


SMP_VERSION = "0.5.0"
SMP_UPSTREAM_COMMIT = "420ce84b0c2df0286fa9bb2bd1499eea625c9b33"


def smp_dependency_sources() -> list[Path]:
    """Identify upstream implementation files without importing the optional library."""
    spec = importlib.util.find_spec("segmentation_models_pytorch")
    if spec is None or spec.origin is None:
        return []
    root = Path(spec.origin).parent
    names = (
        "__version__.py", "decoders/unet/model.py", "decoders/unet/decoder.py",
        "encoders/__init__.py", "encoders/resnet.py", "encoders/_base.py",
        "base/model.py", "base/modules.py", "base/heads.py", "base/initialization.py",
    )
    return [root / name for name in names]


class MarkerSMPUNet(nn.Module):
    """Single-channel DAPI in [0,1] to independent marker intensities in [0,1].

    Uses the upstream ResNet34 encoder and U-Net decoder directly. No external
    weights, auxiliary classification head, decoder attention or input residual.
    ``width=16`` gives SMP's standard decoder channels (256,128,64,32,16).
    """

    def __init__(self, markers: int = 4, width: int = 16,
                 smp_version: str = SMP_VERSION) -> None:
        super().__init__()
        if isinstance(markers, bool) or not isinstance(markers, int) or markers < 1:
            raise ValueError("markers must be a positive integer")
        if isinstance(width, bool) or not isinstance(width, int) or width < 4 or width % 4:
            raise ValueError("width must be a positive multiple of four")
        if smp_version != SMP_VERSION:
            raise ValueError(f"This adapter requires SMP {SMP_VERSION}")
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise ImportError(
                "Install the optional model dependency: "
                "python -m pip install -r requirements-smp-unet.txt"
            ) from error
        if smp.__version__ != SMP_VERSION:
            raise RuntimeError(
                f"Expected segmentation-models-pytorch=={SMP_VERSION}, "
                f"found {smp.__version__}; install requirements-smp-unet.txt"
            )
        self.markers = markers
        self.width = width
        self.network = smp.Unet(
            encoder_name="resnet34", encoder_depth=5, encoder_weights=None,
            in_channels=1, classes=markers, activation=None, aux_params=None,
            decoder_channels=tuple(width * factor for factor in (16, 8, 4, 2, 1)),
            decoder_use_norm="batchnorm", decoder_attention_type=None,
            decoder_interpolation="nearest",
        )

    def forward(self, dapi: torch.Tensor) -> torch.Tensor:
        if not isinstance(dapi, torch.Tensor) or not dapi.is_floating_point():
            raise TypeError("DAPI must be a floating-point tensor")
        if dapi.ndim != 4 or dapi.shape[1] != 1 or dapi.shape[0] < 1:
            raise ValueError("DAPI must have shape N x 1 x H x W")
        height, width = dapi.shape[-2:]
        if min(height, width) < 16:
            raise ValueError("DAPI height and width must each be at least 16")
        # At least 2x2 at the deepest stage keeps BatchNorm valid at batch size 1.
        padded_h = max(64, ((height + 31) // 32) * 32)
        padded_w = max(64, ((width + 31) // 32) * 32)
        if padded_h != height or padded_w != width:
            dapi = F.pad(dapi, (0, padded_w - width, 0, padded_h - height),
                         mode="replicate")
        logits = self.network(dapi)
        return torch.sigmoid(logits[..., :height, :width])
