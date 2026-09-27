"""Try several ensemble configs and pick the best on holdout."""
import argparse, glob, json, os, sys, time, zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, r'E:\aic\final-ihc\74.6531\code')
sys.path.insert(0, r'E:\aic\final-ihc\74.6531\code\src')
from src.data.roi_manifest import MARKERS, PairedMarkers
from src.models.marker_context import MarkerContextNet, local_ssim

CKPT_PATHS = {
    'v6': r'E:\aic\final-ihc\checkpoints\fullplus_cd68_v6\final.pt',
    'v5': r'E:\aic\final-ihc\checkpoints\fullplus_cd68_v5\final.pt',
    'v4': r'E:\aic\final-ihc\checkpoints\fullplus_cd68_v4\final.pt',
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_candidates = glob.glob(r'E:\aic\复赛数据集(包括训练集和测试集输入)')
DATA_ROOT = Path(_candidates[0])
MANIFEST = Path(r'E:\aic\final-ihc\configs\roi_split_semifinal_2026_expanded.json')


def load_model(p):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    cfg = ck['run']['model_config']
    m = MarkerContextNet(width=cfg['width'], markers=cfg['markers'], context=cfg.get('context', True))
    m.load_state_dict(ck.get('ema', ck['model']), strict=True)
    m.eval().to(DEVICE)
    return m


@torch.no_grad()
def predict_tta(model, x, tta=4):
    variants = {1: [(0, False)], 4: [(0, False), (0, True), (2, False), (2, True)],
                8: [(k, f) for k in range(4) for f in (False, True)]}[tta]
    out = None
    for k, f in variants:
        xi = torch.rot90(x, k, (-2, -1))
        if f: xi = xi.flip(-1)
        yi = model(xi).float()
        if f: yi = yi.flip(-1)
        yi = torch.rot90(yi, -k, (-2, -1))
        out = yi if out is None else out + yi
    return out / len(variants)


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    names = manifest['splits']['holdout']
    print(f'[info] holdout samples: {len(names)}')
    ds = PairedMarkers(DATA_ROOT, names, augment=False, cache=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    print('[info] loading models...')
    models = {tag: load_model(p) for tag, p in CKPT_PATHS.items()}

    # Collect predictions per tag, per sample, per marker
    all_preds = {tag: [] for tag in CKPT_PATHS}  # each is a list of (4, H, W)
    all_targets = []
    all_names = []
    t0 = time.time()
    for batch_idx, (x, y, names_) in enumerate(loader):
        x = x.to(DEVICE); y = y.to(DEVICE)
        for tag, m in models.items():
            yp = predict_tta(m, x, 4).cpu()
            for i in range(x.size(0)):
                all_preds[tag].append(yp[i])
        for i in range(x.size(0)):
            all_targets.append(y[i].cpu())
            all_names.append(names_[i])
        if batch_idx % 5 == 0:
            print(f'  [batch {batch_idx}] elapsed={time.time()-t0:.1f}s')
    print(f'[info] preds ready, {len(all_names)} samples')

    # Stack
    P = {tag: torch.stack(all_preds[tag]) for tag in CKPT_PATHS}
    T = torch.stack(all_targets)

    def evaluate(combo):
        # combo: dict of tag -> weight (will be normalized)
        s = sum(combo.values()); combo = {k: v/s for k, v in combo.items()}
        ens = sum(P[tag] * w for tag, w in combo.items())
        ssim_per = {m: [] for m in MARKERS}; psnr_per = {m: [] for m in MARKERS}
        for i in range(len(all_names)):
            for mi, mk in enumerate(MARKERS):
                pi = ens[i, mi:mi+1].clamp(0,1)
                ti = T[i, mi:mi+1]
                with torch.autocast(device_type=DEVICE.type, enabled=False):
                    ssim_per[mk].append(local_ssim(pi.unsqueeze(0).float(), ti.unsqueeze(0).float()).mean().item())
                mse = ((pi-ti)**2).mean().item()
                psnr_per[mk].append(10*np.log10(1.0/max(mse,1e-12)))
        ssim_mean = np.mean([np.mean(v) for v in ssim_per.values()])
        psnr_mean = np.mean([np.mean(v) for v in psnr_per.values()])
        return ssim_mean, psnr_mean, ssim_per, psnr_per

    configs = [
        {'v6': 1.0},
        {'v5': 1.0},
        {'v4': 1.0},
        {'v6': 0.5, 'v5': 0.5},
        {'v6': 0.6, 'v5': 0.4},
        {'v6': 0.7, 'v5': 0.3},
        {'v6': 0.8, 'v5': 0.2},
        {'v6': 0.5, 'v4': 0.5},
        {'v6': 0.6, 'v4': 0.4},
        {'v6': 0.5, 'v5': 0.3, 'v4': 0.2},
        {'v6': 0.4, 'v5': 0.3, 'v4': 0.3},
        {'v6': 0.6, 'v5': 0.3, 'v4': 0.1},
        {'v6': 0.7, 'v5': 0.2, 'v4': 0.1},
    ]
    print()
    print('=== Ensemble comparison ===')
    print(f'{"config":40s}  {"SSIM":>8s}  {"PSNR":>8s}')
    best = (0, None, None)
    for cfg in configs:
        sm, pm, _, _ = evaluate(cfg)
        cstr = '+'.join(f'{k}*{v}' for k,v in cfg.items())
        print(f'{cstr:40s}  {sm:.4f}  {pm:.3f}')
        if sm > best[0]:
            best = (sm, cfg, pm)

    print()
    print(f'BEST: {best[1]}  SSIM={best[0]:.4f}  PSNR={best[2]:.3f}')
    return best, P, T, all_names


if __name__ == '__main__':
    main()
