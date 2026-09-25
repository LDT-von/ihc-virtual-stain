"""Multi-model ensemble inference + per-marker evaluation.

Combines outputs from multiple checkpoints to improve robustness.
"""
import json, sys
from pathlib import Path
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from src.data.roi_manifest import MARKERS, selected_names
from src.models.marker_context import MarkerContextNet, local_ssim
from src.train_marker_context import make_loader, predict

DATA_ROOT = Path(r'E:\aic\复赛数据集(包括训练集和测试集输入)')
MANIFEST = Path(r'E:\aic\final-ihc\configs\roi_split_semifinal_2026_2380.json')
DEVICE = torch.device('cuda')

# Candidate checkpoints for ensemble
CANDIDATES = [
    # (name, path, key='ema')
    ('w96_expanded_baseline', 'checkpoints/ultimate_w96_expanded/baseline/best.pt', 'ema'),
    ('w128_expanded', 'checkpoints/w128_expanded/best.pt', 'ema'),
    ('cd68_aux_v1', 'checkpoints/cd68_aux_v1/best.pt', 'ema'),
    ('cd68_weighted_v2', 'checkpoints/cd68_weighted_w96_v2/best.pt', 'ema'),
]

# Load manifest and get holdout (2380 val = 3 ROIs)
manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
run_src = torch.load('checkpoints/ultimate_w96_expanded/baseline/best.pt', map_location='cpu', weights_only=False)
val_names = selected_names(manifest['splits']['val'], None, run_src['run']['args']['seed'])
loader = make_loader(DATA_ROOT, val_names, 4, cache=True)
print(f"Holdout: {len(val_names)} patches, ROIs: {sorted(set(n.split('_')[0] for n in val_names))}")

def load_model(path, architecture='context'):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    run = ckpt['run']
    if architecture == 'aux':
        from src.models.marker_context_aux import CD68AuxNet
        # Filter config to only known CD68AuxNet kwargs
        known = {'width', 'markers', 'context'}
        filtered = {k: v for k, v in run['model_config'].items() if k in known}
        model = CD68AuxNet(**filtered)
    else:
        model = MarkerContextNet(**run['model_config'])
    model.load_state_dict(ckpt['ema'], strict=True)
    model.to(DEVICE)
    model.eval()
    return model, run

# Individual model evaluation
print("\n=== Individual model scores (TTA=8, 2380 val) ===\n")
models = {}
for name, path, key in CANDIDATES:
    is_aux = 'aux' in name
    arch = 'aux' if is_aux else 'context'
    model, run = load_model(path, architecture=arch)
    models[name] = model
    
    ssim_per = {m: [] for m in MARKERS}
    psnr_per = {m: [] for m in MARKERS}
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if is_aux:
                pred = predict(model, x, tta=8)
            else:
                pred = predict(model, x, tta=8)
            for i, m in enumerate(MARKERS):
                p_i, y_i = pred[:, i:i+1], y[:, i:i+1]
                with torch.autocast(device_type=DEVICE.type, enabled=False):
                    s = local_ssim(p_i.float(), y_i.float()).mean().item()
                mse = ((p_i - y_i) ** 2).mean().item()
                psnr = 10 * np.log10(1.0 / max(mse, 1e-10)) if mse > 0 else 100.0
                ssim_per[m].append(s); psnr_per[m].append(psnr)
    
    s = {m: np.mean(ssim_per[m]) for m in MARKERS}
    p = {m: np.mean(psnr_per[m]) for m in MARKERS}
    avg_s = np.mean(list(s.values()))
    avg_p = np.mean(list(p.values()))
    score = avg_s * 100 + 0.128 * avg_p - 12.85
    print(f"{name:30s} SSIM={avg_s:.4f} PSNR={avg_p:.3f} score={score:.4f}")
    for m in MARKERS:
        print(f"  {m}: SSIM={s[m]:.4f} PSNR={p[m]:.3f}")

# Ensemble evaluation
print("\n=== Ensemble scores (TTA=8, 2380 val) ===\n")
ensemble_sizes = [2, 3, 4]
for n_models in ensemble_sizes:
    if n_models > len(CANDIDATES):
        continue
    # Try all subsets of n_models
    best_score = -1
    best_combo = None
    best_ssim_per = None
    best_psnr_per = None
    
    from itertools import combinations
    for combo in combinations(range(len(CANDIDATES)), n_models):
        combo_names = [CANDIDATES[i][0] for i in combo]
        combo_models = [models[CANDIDATES[i][0]] for i in combo]
        
        ssim_per = {m: [] for m in MARKERS}
        psnr_per = {m: [] for m in MARKERS}
        with torch.no_grad():
            for x, y, _ in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = []
                for m in combo_models:
                    p = predict(m, x, tta=8)
                    preds.append(p)
                # Average predictions
                pred = sum(preds) / len(preds)
                for i, m in enumerate(MARKERS):
                    p_i, y_i = pred[:, i:i+1], y[:, i:i+1]
                    with torch.autocast(device_type=DEVICE.type, enabled=False):
                        s = local_ssim(p_i.float(), y_i.float()).mean().item()
                    mse = ((p_i - y_i) ** 2).mean().item()
                    psnr = 10 * np.log10(1.0 / max(mse, 1e-10)) if mse > 0 else 100.0
                    ssim_per[m].append(s); psnr_per[m].append(psnr)
        
        s = {m: np.mean(ssim_per[m]) for m in MARKERS}
        p = {m: np.mean(psnr_per[m]) for m in MARKERS}
        avg_s = np.mean(list(s.values()))
        avg_p = np.mean(list(p.values()))
        score = avg_s * 100 + 0.128 * avg_p - 12.85
        
        if score > best_score:
            best_score = score
            best_combo = combo_names
            best_ssim_per = s
            best_psnr_per = p
    
    print(f"Top-{n_models} ensemble: score={best_score:.4f}")
    print(f"  Combo: {best_combo}")
    for m in MARKERS:
        print(f"  {m}: SSIM={best_ssim_per[m]:.4f} PSNR={best_psnr_per[m]:.3f}")
