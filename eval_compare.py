# -*- coding: utf-8 -*-
"""Compare multiple checkpoints on holdout split (independent eval, bypasses
audit guard that blocks full-data refit eval).

Usage:
    D:\Anaconda3\python.exe eval_compare.py <ckpt_path> <label> [<ckpt2> <label2> ...]
"""
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, r'E:\aic\final-ihc')

from src.train_marker_context import (
    load_model, make_loader, evaluate, seed_all, write_json, compact,
    SEMIFINAL_SEED,
)

MANIFEST = r"E:\aic\final-ihc\configs\roi_split_semifinal_2026_expanded.json"
DATA_ROOT = r"E:\aic\复赛数据集(包括训练集和测试集输入)"
OUT_DIR = Path(r"E:\aic\final-ihc\analysis")

seed_all(SEMIFINAL_SEED)
device = torch.device('cuda')
manifest = json.loads(Path(MANIFEST).read_text(encoding='utf-8'))
names = sorted(manifest['splits']['holdout'])
print(f"holdout names: {len(names)}")
loader = make_loader(DATA_ROOT, names, 8, augment=False, cache=True)

args = sys.argv[1:]
results = {}
if len(args) < 2 or len(args) % 2 != 0:
    print("Usage: eval_compare.py <ckpt> <label> [<ckpt2> <label2> ...]")
    sys.exit(1)

for ckpt_path, label in zip(args[::2], args[1::2]):
    print(f"\n=== {label}: {ckpt_path} ===")
    sha = hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest()[:12]
    model, _ = load_model(ckpt_path, device)
    t0 = time.perf_counter()
    metrics = evaluate(model, loader, device, tta=4)
    elapsed = time.perf_counter() - t0
    summary = {
        'mean_ssim': metrics['ssim'], 'mean_psnr': metrics['psnr'],
        'per_marker_ssim': {m: metrics['markers'][m]['ssim'] for m in metrics['markers']},
        'per_marker_psnr': {m: metrics['markers'][m]['psnr'] for m in metrics['markers']},
        'elapsed_s': elapsed, 'sha256': sha,
    }
    print(f"  elapsed={elapsed:.1f}s  mean SSIM={summary['mean_ssim']:.4f}  mean PSNR={summary['mean_psnr']:.3f}")
    for m in summary['per_marker_ssim']:
        print(f"  {m:8s}  SSIM={summary['per_marker_ssim'][m]:.4f}  PSNR={summary['per_marker_psnr'][m]:.3f}")
    results[label] = summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / f'compare_{label}.json', metrics)

# Side-by-side table
labels = list(results.keys())
print("\n\n========== SIDE-BY-SIDE ==========")
markers = list(results[labels[0]]['per_marker_ssim'].keys())
header = "metric        " + "".join(f"{l:>16s}" for l in labels)
print(header)
print("-" * len(header))
print("mean_ssim  " + "".join(f"{results[l]['mean_ssim']:>16.4f}" for l in labels))
print("mean_psnr  " + "".join(f"{results[l]['mean_psnr']:>16.3f}" for l in labels))
for m in markers:
    print(f"SSIM {m:6s}" + "".join(f"{results[l]['per_marker_ssim'][m]:>16.4f}" for l in labels))
    print(f"PSNR {m:6s}" + "".join(f"{results[l]['per_marker_psnr'][m]:>16.3f}" for l in labels))

# Delta vs first label
if len(labels) > 1:
    base = labels[0]
    print("\n========== DELTA vs " + base + " ==========")
    print(f"{'metric':<16}" + "".join(f"{l:>16}" for l in labels[1:]))
    print("-" * (16 + 16 * (len(labels) - 1)))
    print(f"{'mean_ssim':<16}" + "".join(f"{results[l]['mean_ssim']-results[base]['mean_ssim']:>+16.4f}" for l in labels[1:]))
    print(f"{'mean_psnr':<16}" + "".join(f"{results[l]['mean_psnr']-results[base]['mean_psnr']:>+16.3f}" for l in labels[1:]))
    for m in markers:
        d_ssim = "".join(f"{results[l]['per_marker_ssim'][m]-results[base]['per_marker_ssim'][m]:>+16.4f}" for l in labels[1:])
        d_psnr = "".join(f"{results[l]['per_marker_psnr'][m]-results[base]['per_marker_psnr'][m]:>+16.3f}" for l in labels[1:])
        print(f"SSIM {m:<10}" + d_ssim)
        print(f"PSNR {m:<10}" + d_psnr)
