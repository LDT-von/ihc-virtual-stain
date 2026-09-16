"""Generate 5 ROI-disjoint folds. Each fold: 20 train ROIs + 5 val ROIs.

Output: configs/roi_split_5fold/fold_{0..4}.json
"""
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.roi_manifest import MARKERS, roi_id, digest, validate_manifest


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='/data1/AIC/data')
    p.add_argument('--out-dir', default='/data1/AIC/configs/roi_split_5fold')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-rois', type=int, default=5)
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in (data_root/'train'/'DAPI').glob('*.jpg'))
    if not names:
        raise ValueError('No training DAPI images')
    groups = sorted({roi_id(n) for n in names})
    n = len(groups)
    if args.val_rois * 5 > n:
        raise ValueError(f'Not enough ROIs ({n}) for 5x{args.val_rois}')

    rng = random.Random(args.seed + 7)
    rng.shuffle(groups)

    summary = []
    for fold_idx in range(5):
        val_set = groups[fold_idx*args.val_rois:(fold_idx+1)*args.val_rois]
        train_set = [g for g in groups if g not in val_set]
        train_names = sorted(n_ for n_ in names if roi_id(n_) in set(train_set))
        val_names = sorted(n_ for n_ in names if roi_id(n_) in set(val_set))

        manifest = {
            'format': 1, 'seed': args.seed, 'fold': fold_idx, 'markers': list(MARKERS),
            'group_unit': 'ROI; 5-fold CV',
            'inventory_names_sha256': digest(sorted(names)),
            'splits': {'train': train_names,
                        'val': val_names,
                        'holdout': []},  # omitted for 5-fold
            'protocol': f'5-fold CV fold {fold_idx+1}/5'
        }
        manifest['sha256'] = digest(manifest)
        validate_manifest(manifest)

        out_path = out_dir / f'fold_{fold_idx}.json'
        out_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        summary.append({
            'fold': fold_idx,
            'train': len(train_names),
            'val': len(val_names),
            'train_rois': sorted(train_set),
            'val_rois': sorted(val_set),
        })
        print(f"Fold {fold_idx}: train={len(train_names)} ({len(train_set)} ROIs)  "
              f"val={len(val_names)} ({len(val_set)} ROIs)  val_ROIs={sorted(val_set)}")

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"\nWrote 5 folds to {out_dir}")


if __name__ == '__main__':
    main()
