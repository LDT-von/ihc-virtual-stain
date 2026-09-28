"""Read-only audit of legacy cached augmentation versus target-preserving V7.

With no data path, a synthetic five-channel patch demonstrates whether repeated
access changes the in-memory cache. With --data-root and --manifest, the same
check samples real training pairs. No images or checkpoints are modified.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.data.paired_clean import CleanPairedMarkers
from src.data.roi_manifest import MARKERS, PairedMarkers, validate_manifest


def make_synthetic(seed):
    rng = np.random.default_rng(seed)
    # Keep most pixels below saturation so accidental highlights remain visible.
    patch = rng.integers(0, 160, size=(1 + len(MARKERS), 256, 256), dtype=np.uint8)
    return ["ROI000_00_00.jpg"], [patch]


def make_real(root, manifest_path, samples, seed):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    names = sorted(sum((manifest["splits"][key] for key in ("train", "val", "holdout")), []))
    if samples < 1:
        raise ValueError("--samples must be positive")
    chosen = sorted(random.Random(seed).sample(names, min(samples, len(names))))
    reader = object.__new__(PairedMarkers)
    reader.root = Path(root)
    return chosen, [reader.read_pair(name) for name in chosen]


def audit(dataset_class, names, arrays, draws, seed):
    dataset = object.__new__(dataset_class)
    dataset.names = list(names)
    dataset.cache = [array.copy() for array in arrays]
    dataset.augment = True
    dataset.aug_strength = 1.0
    if dataset_class is CleanPairedMarkers:
        dataset.policy = "geometry"
    original = [array.copy() for array in dataset.cache]
    random.seed(seed)
    np.random.seed(seed)
    for _ in range(draws):
        for index in range(len(names)):
            dataset[index]

    channel_names = ("DAPI",) + MARKERS
    changed = {}
    for channel, name in enumerate(channel_names):
        diffs = [np.abs(after[channel].astype(np.int16) - before[channel].astype(np.int16))
                 for after, before in zip(dataset.cache, original)]
        changed[name] = {
            "affected_samples": sum(bool(np.any(diff)) for diff in diffs),
            "changed_pixels": sum(int(np.count_nonzero(diff)) for diff in diffs),
            "mean_abs_change": float(np.mean([diff.mean() for diff in diffs])),
            "max_abs_change": max(int(diff.max()) for diff in diffs),
        }
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="Training dataset root; omit for synthetic audit")
    parser.add_argument("--manifest", help="Matching semifinal ROI manifest")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--draws", type=int, default=300,
                        help="Augmented accesses per image, comparable to 300 epochs")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.draws < 1 or bool(args.data_root) != bool(args.manifest):
        parser.error("--draws must be positive; --data-root and --manifest go together")
    mode = "real" if args.data_root else "synthetic"
    names, arrays = (make_real(args.data_root, args.manifest, args.samples, args.seed)
                     if args.data_root else make_synthetic(args.seed))
    legacy = audit(PairedMarkers, names, arrays, args.draws, args.seed)
    clean = audit(CleanPairedMarkers, names, arrays, args.draws, args.seed)
    print(json.dumps({"mode": mode, "sample_count": len(names),
                      "draws_per_sample": args.draws, "legacy_cache_drift": legacy,
                      "v7_geometry_cache_drift": clean}, indent=2))
    if any(value["changed_pixels"] for value in clean.values()):
        raise AssertionError("V7 geometry unexpectedly changed its cache")


if __name__ == "__main__":
    main()
