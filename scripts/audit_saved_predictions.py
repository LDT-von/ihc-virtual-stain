"""Re-score archived JPEG predictions, without claiming held-out provenance."""
import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

MARKERS = ('HLA-DR', 'CD68', 'CD45RO', 'Vimentin')


def score_pair(paths):
    pred_path, target_path = paths
    with Image.open(pred_path) as im:
        pred = np.asarray(im.convert('RGB'))
    with Image.open(target_path) as im:
        target = np.asarray(im.convert('RGB'))
    if pred.shape != target.shape:
        raise ValueError(f'Shape mismatch: {pred_path}')
    # Equivalent to RGB channel-averaged SSIM for triplicated grayscale.
    if (pred[..., :1] == pred).all() and (target[..., :1] == target).all():
        p, t, axis = pred[..., 0], target[..., 0], None
    else:
        p, t, axis = pred, target, -1
    local = structural_similarity(p, t, channel_axis=axis, data_range=255)
    psnr = peak_signal_noise_ratio(t, p, data_range=255)
    x, y = pred.astype(np.float64) / 127.5 - 1, target.astype(np.float64) / 127.5 - 1
    mx, my = x.mean(axis=(0, 1)), y.mean(axis=(0, 1))
    vx, vy = x.var(axis=(0, 1)), y.var(axis=(0, 1))
    cov = ((x-mx)*(y-my)).mean(axis=(0, 1))
    legacy = ((2*mx*my+.02**2)*(2*cov+.06**2)/
              ((mx*mx+my*my+.02**2)*(vx+vy+.06**2)))
    return {'name': target_path.name, 'roi': target_path.stem.split('_')[0],
            'ssim': float(local), 'psnr': float(psnr), 'legacy_global': float(legacy.mean()),
            'prediction_mean': float(pred.mean()), 'target_mean': float(target.mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path('.'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    report = {'protocol': 'skimage default: uniform 7x7 window, sample covariance, data_range=255; RGB channel mean',
              'warning': 'Archived predictions have no checkpoint/split manifest; scores are NOT certified held-out performance.',
              'sets': {}}
    for folder in sorted(args.repo.glob('results_*')):
        result = {}
        for marker in MARKERS:
            predictions = sorted((folder/'val'/marker).glob('*_fake.jpg'))
            if not predictions:
                continue
            pairs = [(f, args.data_root/'train'/marker/(f.stem.removesuffix('_fake')+'.jpg')) for f in predictions]
            missing = [str(t) for _, t in pairs if not t.is_file()]
            if missing:
                raise FileNotFoundError(missing[:5])
            with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
                rows = list(pool.map(score_pair, pairs))
            means = {key: float(np.mean([r[key] for r in rows])) for key in ('ssim', 'psnr', 'legacy_global', 'prediction_mean', 'target_mean')}
            roi_means = {roi: {key: float(np.mean([r[key] for r in rows if r['roi'] == roi])) for key in ('ssim','psnr')} for roi in sorted({r['roi'] for r in rows})}
            result[marker] = dict(count=len(rows), **means, roi_means=roi_means)
            print(folder.name, marker, len(rows), json.dumps(means), flush=True)
        if result:
            report['sets'][folder.name] = result
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
