"""
8x TTA 推理（支持每个 marker 独立 TTA 模式）

用法：
    # 全部 marker 用 8x
    python inference_8x_tta.py --all --split test --output-root results_8xtta --batch-size 4
    
    # 全部 marker 用 4x（仅翻转）
    python inference_8x_tta.py --all --split val --tta-mode 4x --output-root results_4xtta
"""
import argparse
import time
from pathlib import Path
import sys
import random
from torch.utils.data import DataLoader, Subset

import torch

ROOT = Path(r'E:\aic\ihc-virtual-stain')
sys.path.insert(0, str(ROOT))
from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import MetricAggregator, to_uint8
from src.models.pix2pix_gan import build_pix2pix_model


def find_latest_ckpt(marker: str):
    """找最新 checkpoint"""
    ck_root = ROOT / 'checkpoints'
    if not ck_root.exists():
        return None
    dirs = sorted(
        (d for d in ck_root.glob(f'pix2pix_v2_{marker}_*') if d.is_dir()),
        key=lambda d: int(d.name.rsplit('_', 1)[1]),
        reverse=True,
    )
    for d in dirs:
        if (d / 'best.pt').exists():
            return d / 'best.pt'
        epoch_files = sorted(d.glob('epoch*.pt'), key=lambda p: int(p.stem[5:]), reverse=True)
        if epoch_files:
            return epoch_files[0]
        if (d / 'final.pt').exists():
            return d / 'final.pt'
    return None


def tta_4x_forward(model, dapi: torch.Tensor) -> torch.Tensor:
    """4x TTA：4 种翻转平均"""
    outputs = []
    for flip_code in range(4):
        if flip_code == 0:
            d = dapi
        elif flip_code == 1:
            d = torch.flip(dapi, dims=(3,))
        elif flip_code == 2:
            d = torch.flip(dapi, dims=(2,))
        else:
            d = torch.flip(dapi, dims=(2, 3))
        out = model.generator(d, d)
        if flip_code == 1:
            out = torch.flip(out, dims=(3,))
        elif flip_code == 2:
            out = torch.flip(out, dims=(2,))
        elif flip_code == 3:
            out = torch.flip(out, dims=(2, 3))
        outputs.append(out)
    return torch.stack(outputs).mean(dim=0)


def tta_8x_forward(model, dapi: torch.Tensor) -> torch.Tensor:
    """Eight unique D4 transforms, with both flip and rotation inverted."""
    outputs = []
    for flip in (False, True):
        d = torch.flip(dapi, dims=(3,)) if flip else dapi
        for rot in range(4):
            d_rot = torch.rot90(d, rot, dims=(2, 3))
            out = model.generator(d_rot, d_rot)
            out = torch.rot90(out, -rot, dims=(2, 3))
            if flip:
                out = torch.flip(out, dims=(3,))
            outputs.append(out)
    return torch.stack(outputs).mean(dim=0)


def run_inference(marker: str, ckpt_path: Path, data_root: Path,
                   output_dir: Path, batch_size: int, device: torch.device,
                   compute_metrics: bool, split: str = 'val', 
                   val_split: float = 0.05, tta_mode: str = '8x'):
    """对单个 marker 运行 TTA 推理
    
    tta_mode: '8x' = 2 flip states x 4 rotations; '4x' = 4 flips
    """
    print(f"\n{'='*60}")
    print(f"[TTA-{tta_mode}] marker={marker} split={split}")
    print(f"[TTA-{tta_mode}] ckpt={ckpt_path}")
    
    model = build_pix2pix_model(
        input_channels=3, cond_channels=3, output_channels=3, base_filters=64
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    forward_fn = tta_8x_forward if tta_mode == '8x' else tta_4x_forward
    
    if split == 'val':
        full_ds = DAPItoIHCDataset(
            root=data_root, marker=marker, split='train',
            patch_size=256, augment=False,
        )
        n = len(full_ds)
        val_n = max(1, int(n * val_split))
        val_n = min(val_n, n - 1)
        rng = random.Random(42)
        val_idx = sorted(rng.sample(range(n), val_n))
        dataset = Subset(full_ds, val_idx)
    else:
        dataset = DAPItoIHCDataset(
            root=data_root, marker=marker, split=split,
            patch_size=256, augment=False,
        )
    
    if len(dataset) == 0:
        print(f"[TTA] {marker} {split}: 数据集为空")
        return None
    
    print(f"[Dataset] {split}/{marker}: {len(dataset)} 张")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)
    
    metric = MetricAggregator() if compute_metrics else None
    written = 0
    start = time.time()
    
    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device, non_blocking=True)
            fake = forward_fn(model, dapi)
            
            if metric is not None:
                metric.update(fake, batch['ihc'].to(device))
            
            for i, name in enumerate(batch['name']):
                from PIL import Image
                img = Image.fromarray(to_uint8(fake[i].cpu()))
                img.save(output_dir / f'{name}_fake.jpg', format='JPEG', quality=95)
                written += 1
    
    elapsed = time.time() - start
    if written > 0:
        print(f"[TTA-{tta_mode}] {marker}: {written} 张，{elapsed:.1f}s")
    
    if metric is not None:
        r = metric.result()
        print(f"[TTA-{tta_mode}] {marker} SSIM={r['ssim']:.4f} PSNR={r['psnr']:.2f}")
        return r
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--marker', type=str, default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--data-root',
                   default='E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
    p.add_argument('--output-root', default='E:/aic/ihc-virtual-stain/results_tta')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    p.add_argument('--val-split', type=float, default=0.05)
    p.add_argument('--tta-mode', default='8x', choices=['4x', '8x'],
                   help='TTA 模式：4x 仅翻转; 8x 翻转+旋转')
    args = p.parse_args()
    
    markers = [args.marker] if args.marker else ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']
    if args.all:
        markers = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    
    results = {}
    for marker in markers:
        ckpt = Path(args.ckpt) if args.ckpt else find_latest_ckpt(marker)
        if ckpt is None or not ckpt.exists():
            print(f"[TTA] {marker}: 找不到 checkpoint，跳过")
            continue
        
        output_dir = output_root / split_part(args.split) / marker
        output_dir.mkdir(parents=True, exist_ok=True)
        
        r = run_inference(
            marker=marker, ckpt_path=ckpt, data_root=data_root,
            output_dir=output_dir, batch_size=args.batch_size,
            device=device, compute_metrics=args.split != 'test', split=args.split,
            val_split=args.val_split, tta_mode=args.tta_mode,
        )
        if r:
            results[marker] = r
    
    print(f"\n{'='*60}")
    print(f"{args.tta_mode} TTA 推理结果汇总：")
    total_ssim = 0.0
    for marker, r in results.items():
        print(f"  {marker}: SSIM={r['ssim']:.4f} PSNR={r['psnr']:.2f}")
        total_ssim += r['ssim']
    
    if results:
        avg = total_ssim / len(results)
        print(f"\n  平均 SSIM: {avg:.4f}")
        print("  此为本地 SSIM，不是平台综合分（PSNR 归一化公式未确认）")
    print(f"  结果目录: {output_root}")
    print('='*60)


def split_part(s):
    return s


if __name__ == '__main__':
    main()
