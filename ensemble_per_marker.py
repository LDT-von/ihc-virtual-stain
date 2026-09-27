"""Per-marker ensemble weight search. Sometimes one model excels on CD68 while another excels on CD45RO."""
import json, glob, sys, time
from pathlib import Path
import numpy as np
import torch
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
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    names = manifest['splits']['holdout']
    print(f'[info] holdout: {len(names)}')
    ds = PairedMarkers(DATA_ROOT, names, augment=False, cache=True)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    print('[info] loading models...')
    models = {tag: load_model(p) for tag, p in CKPT_PATHS.items()}
    P = {tag: [] for tag in CKPT_PATHS}
    T = []; Nm = []
    t0 = time.time()
    for bi, (x, y, names_) in enumerate(loader):
        x = x.to(DEVICE); y = y.to(DEVICE)
        for tag, m in models.items():
            yp = predict_tta(m, x, 4).cpu()
            for i in range(x.size(0)):
                P[tag].append(yp[i])
        for i in range(x.size(0)):
            T.append(y[i].cpu()); Nm.append(names_[i])
        if bi % 5 == 0:
            print(f'  [batch {bi}] elapsed={time.time()-t0:.1f}s')
    P = {k: torch.stack(v) for k, v in P.items()}
    T = torch.stack(T)

    def eval_per_marker(weights):
        # weights: dict tag -> weight, summed per marker
        ws = sum(weights.values()); weights = {k: v/ws for k, v in weights.items()}
        ens = sum(P[tag] * weights.get(tag, 0) for tag in P)
        out = {}
        for mi, mk in enumerate(MARKERS):
            s = []; p = []
            for i in range(len(Nm)):
                pi = ens[i, mi:mi+1].clamp(0,1); ti = T[i, mi:mi+1]
                with torch.autocast(device_type=DEVICE.type, enabled=False):
                    s.append(local_ssim(pi.unsqueeze(0).float(), ti.unsqueeze(0).float()).mean().item())
                mse = ((pi-ti)**2).mean().item()
                p.append(10*np.log10(1.0/max(mse,1e-12)))
            out[mk] = (np.mean(s), np.mean(p))
        return out

    print()
    print('=== Per-marker best individual ===')
    for tag in CKPT_PATHS:
        r = eval_per_marker({tag: 1.0})
        print(f'  {tag}:')
        for mk in MARKERS:
            print(f'    {mk:10s}  SSIM={r[mk][0]:.4f}  PSNR={r[mk][1]:.3f}')

    print()
    print('=== Per-marker best 2-model ensemble ===')
    best_per = {}
    for mi, mk in enumerate(MARKERS):
        best = (0, None)
        # Try all 2-tag combos with 7 weight splits
        for t1 in ['v6','v5','v4']:
            for t2 in ['v6','v5','v4']:
                if t1 >= t2: continue
                for w1 in np.arange(0.1, 1.0, 0.1):
                    r = eval_per_marker({t1: w1, t2: 1-w1})
                    if r[mk][0] > best[0]:
                        best = (r[mk][0], {t1: w1, t2: 1-w1})
        best_per[mk] = best
        print(f'  {mk}: SSIM={best[0]:.4f}  weights={best[1]}')

    # Apply per-marker best to final ensemble
    print()
    print('=== Per-marker weighted ensemble ===')
    final = torch.zeros_like(P['v6'])
    weights_used = {}
    for mi, mk in enumerate(MARKERS):
        ws = best_per[mk][1]
        ws_norm = {k: v/sum(ws.values()) for k, v in ws.items()}
        weights_used[mk] = ws_norm
        for tag, w in ws_norm.items():
            final[:, mi:mi+1] += P[tag][:, mi:mi+1] * w
    ssim_all = []; psnr_all = []
    for i in range(len(Nm)):
        for mi, mk in enumerate(MARKERS):
            pi = final[i, mi:mi+1].clamp(0,1); ti = T[i, mi:mi+1]
            with torch.autocast(device_type=DEVICE.type, enabled=False):
                ssim_all.append(local_ssim(pi.unsqueeze(0).float(), ti.unsqueeze(0).float()).mean().item())
            mse = ((pi-ti)**2).mean().item()
            psnr_all.append(10*np.log10(1.0/max(mse,1e-12)))
    ssim_mean = np.mean(ssim_all); psnr_mean = np.mean(psnr_all)
    print(f'  Per-marker tuned ensemble:  SSIM={ssim_mean:.4f}  PSNR={psnr_mean:.3f}')
    print(f'  Weights:')
    for mk in MARKERS:
        print(f'    {mk}: {weights_used[mk]}')

    # Compare to v6 alone
    r6 = eval_per_marker({'v6': 1.0})
    print()
    print(f'v6 alone:  SSIM={np.mean([r6[m][0] for m in MARKERS]):.4f}')
    print(f'Delta:     {ssim_mean - np.mean([r6[m][0] for m in MARKERS]):+.4f}')
    return weights_used, ssim_mean, psnr_mean


if __name__ == '__main__':
    main()
