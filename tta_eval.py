"""TTA 评估脚本：使用真正的 val 集评估 TTA 提升

用法：
    python tta_eval.py --marker HLA-DR --tta-mode 8x
"""
import argparse
import time
from pathlib import Path
import sys
from torch.utils.data import DataLoader

import torch

ROOT = Path(r'E:\aic\ihc-virtual-stain')
sys.path.insert(0, str(ROOT))
from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import MetricAggregator, to_uint8
from src.models.pix2pix_gan import build_pix2pix_model

# 4 个 marker 的最佳 checkpoint
BEST_CKPTS = {
    'HLA-DR':   ROOT / 'checkpoints' / 'pix2pix_v2_HLA-DR_1788962683' / 'best.pt',
    'CD68':     ROOT / 'checkpoints' / 'pix2pix_v2_CD68_1788969904' / 'best.pt',
    'CD45RO':   ROOT / 'checkpoints' / 'pix2pix_v7_CD45RO_1789200177' / 'best.pt',
    'Vimentin': ROOT / 'checkpoints' / 'pix2pix_v7_Vimentin_1789200192' / 'best.pt',
}


def tta_4x(model, dapi):
    outs = []
    for c in range(4):
        if c == 0: d = dapi
        elif c == 1: d = torch.flip(dapi, dims=(3,))
        elif c == 2: d = torch.flip(dapi, dims=(2,))
        else: d = torch.flip(dapi, dims=(2, 3))
        out = model.generator(d, d)
        if c == 1: out = torch.flip(out, dims=(3,))
        elif c == 2: out = torch.flip(out, dims=(2,))
        elif c == 3: out = torch.flip(out, dims=(2, 3))
        outs.append(out)
    return torch.stack(outs).mean(dim=0)


def tta_8x(model, dapi):
    from inference_8x_tta import tta_8x_forward
    return tta_8x_forward(model, dapi)


def evaluate(marker, ckpt, tta_mode, batch_size, device, data_root):
    print(f"\n[{tta_mode}] {marker}")
    model = build_pix2pix_model(3, 3, 3, 64).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()

    ds = DAPItoIHCDataset(root=data_root, marker=marker, split='val',
                          patch_size=256, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    forward = tta_8x if tta_mode == '8x' else tta_4x
    metric = MetricAggregator()
    t0 = time.time()
    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device)
            fake = forward(model, dapi)
            metric.update(fake, batch['ihc'].to(device))
    r = metric.result()
    print(f"[{tta_mode}] {marker}: SSIM={r['ssim']:.4f} PSNR={r['psnr']:.2f} ({time.time()-t0:.1f}s)")
    del model
    torch.cuda.empty_cache()
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--marker', default='HLA-DR')
    p.add_argument('--tta-mode', default='8x', choices=['4x', '8x'])
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--data-root', default='E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = BEST_CKPTS[args.marker]
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)
    evaluate(args.marker, ckpt, args.tta_mode, args.batch_size, device, Path(args.data_root))


if __name__ == '__main__':
    main()
