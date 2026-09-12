"""v7 测试集推理脚本 - 4 marker 单次推理

4 个 marker 的最佳 checkpoint：
- HLA-DR:   pix2pix_v6_HLA-DR_1789185490/best.pt
- CD68:     pix2pix_v6_CD68_1789186851/best.pt
- CD45RO:   pix2pix_v7_CD45RO_1789200177/best.pt (新!)
- Vimentin: pix2pix_v7_Vimentin_1789200192/best.pt (新!)

不带 TTA - 单次推理确保色彩一致性
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

# 4 个 marker 的最佳 checkpoint（基于真实验证集评估）
BEST_CKPTS = {
    'HLA-DR':   ROOT / 'checkpoints' / 'pix2pix_v2_HLA-DR_1788962683' / 'best.pt',
    'CD68':     ROOT / 'checkpoints' / 'pix2pix_v2_CD68_1788969904' / 'best.pt',
    'CD45RO':   ROOT / 'checkpoints' / 'pix2pix_v7_CD45RO_1789200177' / 'best.pt',
    'Vimentin': ROOT / 'checkpoints' / 'pix2pix_v7_Vimentin_1789200192' / 'best.pt',
}

for marker, ckpt in BEST_CKPTS.items():
    if not ckpt.exists():
        print(f'[MISS] {marker}: {ckpt} NOT FOUND')
        sys.exit(1)
    print(f'[OK] {marker}: {ckpt.name}')


def run_inference(marker, ckpt_path, data_root, output_dir, batch_size, device, split):
    print(f"\n{'='*60}")
    print(f"[v7 single] marker={marker} split={split}")
    print(f"[v7 single] ckpt={ckpt_path}")

    model = build_pix2pix_model(
        input_channels=3, cond_channels=3, output_channels=3, base_filters=64
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    if split == 'val':
        ds = DAPItoIHCDataset(
            root=data_root, marker=marker, split='val',
            patch_size=256, augment=False,
        )
    else:
        ds = DAPItoIHCDataset(
            root=data_root, marker=marker, split='test',
            patch_size=256, augment=False,
        )

    print(f"[Dataset] {split}/{marker}: {len(ds)} 张")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

    metric = MetricAggregator() if split == 'val' else None
    written = 0
    start = time.time()

    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device, non_blocking=True)
            # 单次推理，不做 TTA
            fake = model.generator(dapi, dapi)

            if metric is not None and 'ihc' in batch:
                metric.update(fake, batch['ihc'].to(device))

            for i, name in enumerate(batch['name']):
                from PIL import Image
                img = Image.fromarray(to_uint8(fake[i].cpu()))
                img.save(output_dir / f'{name}_fake.jpg', format='JPEG', quality=95)
                written += 1

    elapsed = time.time() - start
    print(f"[v7 single] {marker}: {written} 张，{elapsed:.1f}s")

    if metric is not None:
        r = metric.result()
        print(f"[v7 single] {marker} SSIM={r['ssim']:.4f} PSNR={r['psnr']:.2f}")
        return r
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--output-root', default='E:/aic/ihc-virtual-stain/results_v7_test')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--data-root',
                       default='E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
    parser.add_argument('--marker', default=None, help='Single marker (default: all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    markers = [args.marker] if args.marker else ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']

    results = {}
    for marker in markers:
        ckpt = BEST_CKPTS[marker]
        output_dir = output_root / args.split / marker
        r = run_inference(
            marker=marker, ckpt_path=ckpt, data_root=data_root,
            output_dir=output_dir, batch_size=args.batch_size,
            device=device, split=args.split,
        )
        if r:
            results[marker] = r

    if args.split == 'val' and results:
        print(f"\n{'='*60}")
        print(f"v7 单次推理 验证集结果：")
        total_ssim = 0.0
        for marker, r in results.items():
            print(f"  {marker}: SSIM={r['ssim']:.4f} PSNR={r['psnr']:.2f}")
            total_ssim += r['ssim']
        avg = total_ssim / len(results)
        print(f"\n  平均 SSIM: {avg:.4f}")
        print(f"  对应平台分: {avg * 100:.2f}")
        print(f"  结果目录: {output_root}")
        print('='*60)


if __name__ == '__main__':
    main()
