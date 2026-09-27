"""Final 'all-in' ensemble: combine all useful ckpts.

Rationale: holdout SSIM doesn't predict platform SSIM (v3 was best on platform
at 75.11 but only 0.7926 on holdout). So we ensemble across architectures/stages
to cover both regimes: high-holdout models (v6) and historically platform-strong
ones (v3 lineage + w96 fine-tunes).

Weights are tuned to balance the two regimes:
  v6:           0.30   (high-holdout)
  v3:           0.10   (platform-best, low-holdout)
  v3p1:         0.10   (v3 follow-up)
  v3_continue:  0.10   (v3 follow-up)
  w96_balanced: 0.20   (w96 lineage, medium)
  w96_vim_boost:0.20   (w96 lineage, vimentin-boost)
"""
import argparse, glob, json, sys, time, zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, r'E:\aic\final-ihc\74.6531\code')
sys.path.insert(0, r'E:\aic\final-ihc\74.6531\code\src')
from src.data.roi_manifest import MARKERS, PairedMarkers
from src.models.marker_context import MarkerContextNet, local_ssim

CKPTS = {
    'v6':           (r'E:\aic\final-ihc\checkpoints\fullplus_cd68_v6\final.pt',          0.30),
    'v3':           (r'E:\aic\final-ihc\checkpoints\fullplus_cd68_v3\final.pt',          0.10),
    'v3p1':         (r'E:\aic\final-ihc\checkpoints\fullplus_cd68_v3p1\final.pt',        0.10),
    'v3_continue':  (r'E:\aic\final-ihc\checkpoints\fullplus_cd68_v3_continue\final.pt', 0.10),
    'w96_balanced': (r'E:\aic\final-ihc\checkpoints\w96_balanced\final.pt',              0.20),
    'w96_vim_boost':(r'E:\aic\final-ihc\checkpoints\w96_vim_boost\final.pt',             0.20),
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = Path(glob.glob(r'E:\aic\复赛数据集(包括训练集和测试集输入)')[0])
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
    variants = {1: [(0, False)], 4: [(0, False), (0, True), (2, False), (2, True)]}[tta]
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--tta', type=int, default=4)
    args = ap.parse_args()

    out_zip = Path(args.out)
    assert not out_zip.exists(), f'Will not overwrite {out_zip}'

    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    names = manifest['splits']['holdout']
    print(f'[info] holdout: {len(names)}')
    ds = PairedMarkers(DATA_ROOT, names, augment=False, cache=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    # Load all models
    models = {}
    for tag, (p, w) in CKPTS.items():
        print(f'[info] loading {tag} (w={w})')
        models[tag] = (load_model(p), w)
    ws = sum(w for _, w in models.values())
    weights = {tag: w/ws for tag, (_, w) in models.items()}
    print('[info] normalized weights:', {k: round(v, 3) for k, v in weights.items()})

    # Output dir
    tmp = out_zip.parent / ('_ensall_tmp_' + out_zip.stem)
    if tmp.exists():
        import shutil; shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for marker in MARKERS:
        (tmp / marker).mkdir(parents=True)

    ssim_per = {m: [] for m in MARKERS}
    psnr_per = {m: [] for m in MARKERS}
    t0 = time.time()
    for bi, (x, y, names_) in enumerate(loader):
        x = x.to(DEVICE); y = y.to(DEVICE)
        ens = None
        for tag, (m, _) in models.items():
            yp = predict_tta(m, x, args.tta)
            ens = (yp * weights[tag]) if ens is None else ens + yp * weights[tag]
        for i in range(x.size(0)):
            name = names_[i]
            for mi, mk in enumerate(MARKERS):
                pi = ens[i, mi:mi+1].clamp(0,1); ti = y[i, mi:mi+1]
                with torch.autocast(device_type=DEVICE.type, enabled=False):
                    ssim_per[mk].append(local_ssim(pi.unsqueeze(0).float(), ti.unsqueeze(0).float()).mean().item())
                mse = ((pi-ti)**2).mean().item()
                psnr_per[mk].append(10*np.log10(1.0/max(mse,1e-12)))
                arr = (pi[0].clamp(0,1) * 255).round().to(torch.uint8).cpu().numpy()
                Image.fromarray(arr, mode='L').save(tmp / mk / name, format='JPEG', quality=100, subsampling=0)
        if bi % 5 == 0:
            print(f'  [batch {bi}] elapsed={time.time()-t0:.1f}s')

    print()
    print('=== ALL-IN ensemble holdout ===')
    print(f'{"marker":10s}  {"SSIM":>8s}  {"PSNR":>8s}')
    for mk in MARKERS:
        ms = np.mean(ssim_per[mk]); mp = np.mean(psnr_per[mk])
        print(f'{mk:10s}  {ms:.4f}  {mp:.3f}')
    ssim_mean = np.mean([np.mean(ssim_per[m]) for m in MARKERS])
    psnr_mean = np.mean([np.mean(psnr_per[m]) for m in MARKERS])
    print(f'{"mean":10s}  {ssim_mean:.4f}  {psnr_mean:.3f}')

    # Zip
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for mk in MARKERS:
            for p in sorted((tmp / mk).glob('*.jpg')):
                zf.write(p, arcname=f'{mk}/{p.name}')
    print(f'[ok] wrote {out_zip} ({out_zip.stat().st_size / 1e6:.1f} MB)')

    # Manifest
    out_zip.with_suffix('.manifest.json').write_text(json.dumps({
        'ensemble': list(CKPTS.keys()),
        'raw_weights': {k: v for k, (_, v) in CKPTS.items()},
        'normalized_weights': weights,
        'tta': args.tta,
        'holdout': {
            'per_marker': {m: {'ssim': float(np.mean(ssim_per[m])), 'psnr': float(np.mean(psnr_per[m]))} for m in MARKERS},
            'mean_ssim': float(ssim_mean),
            'mean_psnr': float(psnr_mean),
        }
    }, indent=2))
    print('[ok] wrote manifest')


if __name__ == '__main__':
    main()
