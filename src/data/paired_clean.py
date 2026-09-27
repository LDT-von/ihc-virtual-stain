"""Paired DAPI/marker data with target-preserving augmentation."""

import math
import random

import numpy as np
import torch

from .roi_manifest import PairedMarkers


class CleanPairedMarkers(PairedMarkers):
    """Apply synchronized geometry and, optionally, noise to DAPI alone.

    ``read_pair`` and the optional uint8 cache come from ``PairedMarkers``.
    Every sample is copied before transformation, so cached pixels and marker
    targets retain their original intensities.
    """

    def __init__(
        self,
        root,
        names,
        augment=False,
        cache=True,
        policy="geometry",
        aug_strength=1.0,
    ):
        if policy not in ("geometry", "dapi-noise"):
            raise ValueError(f"Unknown augmentation policy: {policy}")
        strength = float(aug_strength)
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("aug_strength must be finite and nonnegative")
        if policy == "dapi-noise" and strength != 0 and not 0.2 <= strength <= 2.5:
            raise ValueError(
                "dapi-noise requires aug_strength=0 or 0.2 <= aug_strength <= 2.5 "
                "so the sigma interval and probability remain valid"
            )

        self.policy = policy
        super().__init__(
            root,
            names,
            augment=augment,
            cache=cache,
            aug_strength=strength,
        )

    def __getitem__(self, index):
        name = self.names[index]
        arr = self.cache[index] if self.cache is not None else self.read_pair(name)

        if self.augment:
            # A single transform is shared by DAPI and every marker.
            arr = np.rot90(arr, random.randrange(4), axes=(-2, -1))
            if random.random() < 0.5:
                arr = arr[..., ::-1]

        # A real copy is required: rotations/flips may return cache-backed
        # views (including negative strides), and the target must stay clean.
        values = np.array(arr, dtype=np.float32, copy=True)
        values *= np.float32(1.0 / 255.0)
        x = values[:1].copy()
        y = values[1:].copy()

        if self.augment and self.policy == "dapi-noise":
            if random.random() < 0.4 * self.aug_strength:
                sigma = random.uniform(1.0, 5.0) * self.aug_strength / 255.0
                noise = np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)
                x = np.clip(x + noise, 0.0, 1.0).astype(np.float32, copy=False)

        return torch.from_numpy(x), torch.from_numpy(y), name
