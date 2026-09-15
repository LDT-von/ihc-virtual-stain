"""Train-only ridge baseline: DAPI intensity and local context, no neural network.

This is a sanity baseline, NOT a replacement for a matched Pix2Pix experiment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.roi_manifest import MARKERS, read_gray, roi_id
from src.train_marker_context import load_manifest


def features(image):
    x = image.astype(np.float32)/255
    return np.stack((np.ones_like(x), x, x*x, np.sqrt(x),
                     uniform_filter(x, 5), uniform_filter(x, 15),
                     uniform_filter(x, 31), np.full_like(x, x.mean())), axis=-1)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = load_manifest(args.manifest, args.data_root)

    def train_statistics(name):
        feat = features(read_gray(args.data_root/'train'/'DAPI'/name)).reshape(-1, 8)
        target = np.stack([read_gray(args.data_root/'train'/m/name).ravel() for m in MARKERS], -1)/255
        # Deterministic uniformly spread subset of pixels, from training ROIs only.
        x, y = feat[::257].astype(np.float64), target[::257].astype(np.float64)
        return x.T@x, x.T@y

    xtx, xty = np.zeros((8, 8)), np.zeros((8, 4))
    with ThreadPoolExecutor(4) as pool:
        for a, b in pool.map(train_statistics, manifest['splits']['train']):
            xtx += a
            xty += b
    coefficients = np.linalg.solve(xtx+np.eye(8)*1., xty)

    def evaluate_image(name):
        feat = features(read_gray(args.data_root/'train'/'DAPI'/name))
        pred = np.rint(np.clip(feat@coefficients, 0, 1)*255).astype(np.uint8)
        rows = {}
        for i, m in enumerate(MARKERS):
            target = read_gray(args.data_root/'train'/m/name)
            rows[m] = {'ssim': float(structural_similarity(target, pred[..., i], data_range=255)),
                       'psnr': float(peak_signal_noise_ratio(target, pred[..., i], data_range=255))}
        return {'name': name, 'roi': roi_id(name), 'markers': rows}

    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(evaluate_image, manifest['splits']['val']))
    markers = {m: {k: float(np.mean([r['markers'][m][k] for r in rows])) for k in ('ssim', 'psnr')} for m in MARKERS}
    result = {'method': 'train-only 8-feature ridge sanity baseline', 'count': len(rows),
              'split_sha256': manifest['sha256'], 'coefficients': coefficients.tolist(),
              'markers': markers, 'ssim': float(np.mean([m['ssim'] for m in markers.values()])),
              'psnr': float(np.mean([m['psnr'] for m in markers.values()])),
              'warning': 'Not a matched neural baseline; does not establish superiority to Pix2Pix.', 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
