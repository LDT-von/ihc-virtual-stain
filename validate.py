"""验证集评估脚本：评估模型在验证集上的 SSIM/PSNR

用法：
    python validate.py --marker HLA-DR --ckpt checkpoints/xxx/best.pt
"""
import argparse
import builtins
import time
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DAPItoIHCDataset
from src.models.pix2pix_gan import build_pix2pix_model
from src.models.losses import PyramidL1SSIMLoss
from src.metrics.ssim_psnr import MetricAggregator, to_uint8

# GBK 安全打印
_orig_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    try:
        _orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        safe = [str(a).encode('gbk', errors='replace').decode('gbk') for a in args]
        _orig_print(*safe, **kwargs)
builtins.print = _flush_print


def parse_args():
    p = argparse.ArgumentParser(description="验证集评估")
    p.add_argument("--marker", type=str, default="HLA-DR")
    p.add_argument("--ckpt", type=str, required=True, help="模型 checkpoint 路径")
    p.add_argument("--data_root", type=str,
                   default="E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default=None, help="可视化输出目录")
    return p.parse_args()


def compute_metrics(fake, real):
    """计算 MSE"""
    mse = ((fake - real) ** 2).mean().item()
    return mse


def main():
    args = argparse.ArgumentParser()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=" * 60)
    print(f"[Validate] marker={args.marker}")
    print(f"[Validate] checkpoint={args.ckpt}")
    print(f"[Validate] device={device}")
    print(f"=" * 60)
    
    # 加载验证集
    val_ds = DAPItoIHCDataset(
        root=args.data_root,
        marker=args.marker,
        split="val",
        patch_size=256,
        augment=False,
    )
    
    if len(val_ds) == 0:
        print("[Error] 验证集为空！请先运行 split_data.py 划分数据")
        return
    
    print(f"[Dataset] 验证集: {len(val_ds)} samples")
    
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    
    # 加载模型
    model = build_pix2pix_model(
        input_channels=3, cond_channels=3, output_channels=3, base_filters=64,
    ).to(device)
    
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"[Model] loaded checkpoint")
    
    # 评估
    model.eval()
    aggregator = MetricAggregator()
    mse_list = []
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            dapi = batch["dapi"].to(device, non_blocking=True)
            real_img = batch["ihc"].to(device, non_blocking=True)
            names = batch["name"]
            
            fake_img = model.generator(dapi, dapi)
            
            for j in range(fake_img.shape[0]):
                mse = compute_metrics(fake_img[j:j+1], real_img[j:j+1])
                mse_list.append(mse)
            
            aggregator.update(fake_img, real_img)
            
            if (i + 1) % 20 == 0:
                print(f"[Validate] {i+1}/{len(val_loader)} ...")
    
    elapsed = time.time() - t0
    
    # 统计
    metrics = aggregator.result()
    ssim_mean = metrics["ssim"]
    psnr_mean = metrics["psnr"]
    mse_mean = float(np.mean(mse_list))
    
    print(f"\n{'='*60}")
    print(f"[结果] marker={args.marker}")
    print(f"{'='*60}")
    print(f"  SSIM  = {ssim_mean:.4f}")
    print(f"  PSNR  = {psnr_mean:.2f} dB")
    print(f"  MSE   = {mse_mean:.6f}")
    print(f"  时间  = {elapsed:.1f}s ({elapsed/len(val_ds)*1000:.1f}ms/张)")
    print(f"{'='*60}")
    
    # 保存结果
    results_path = Path(args.ckpt).parent / f"val_results_{args.marker}.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"marker={args.marker}\n")
        f.write(f"ckpt={args.ckpt}\n")
        f.write(f"SSIM={ssim_mean:.4f}\n")
        f.write(f"PSNR={psnr_mean:.2f}\n")
        f.write(f"MSE={mse_mean:.6f}\n")
    print(f"[保存] 结果已保存到 {results_path}")


if __name__ == "__main__":
    main()
