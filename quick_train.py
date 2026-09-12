"""快速训练脚本：使用验证集评估，只训练几个 epoch

用法：
    python quick_train.py --marker HLA-DR --epochs 5 --batch_size 16
"""
import argparse
import sys
import builtins
import time
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import random

# 强制 unbuffered output
_orig_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    try:
        _orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        # PowerShell 默认 GBK，去掉特殊字符
        safe = [str(a).encode('gbk', errors='replace').decode('gbk') for a in args]
        _orig_print(*safe, **kwargs)
builtins.print = _flush_print

from src.data.dataset import DAPItoIHCDataset
from src.models.pix2pix_gan import build_pix2pix_model
from src.models.losses import PyramidL1SSIMLoss, GANLoss
from src.metrics.ssim_psnr import MetricAggregator


def parse_args():
    p = argparse.ArgumentParser(description="快速训练 + 验证")
    p.add_argument("--data_root", type=str,
                   default="E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）")
    p.add_argument("--marker", type=str, default="HLA-DR")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--pyramid_levels", type=int, default=4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_every", type=int, default=1, help="每隔几个 epoch 验证一次")
    p.add_argument("--resume_ckpt", type=str, default=None)
    p.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例，剩余做验证")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=" * 60)
    print(f"[Quick Train] marker={args.marker}")
    print(f"[Quick Train] epochs={args.epochs} bs={args.batch_size}")
    print(f"[Quick Train] device={device}")
    print(f"=" * 60)
    
    # ====== 加载训练集 + 排除验证集 ======
    full_train_ds = DAPItoIHCDataset(
        root=args.data_root,
        marker=args.marker,
        split="train",
        patch_size=256,
        augment=True,
    )

    # ====== 加载验证集（已划分好的）======
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

    # 训练时排除验证集里已经分走的文件（按文件名前缀/名称去重）
    val_names = set()
    for ds_item in val_ds.pairs:
        if ds_item[0] is not None:
            val_names.add(ds_item[0].name)
    # 重建训练集：排除验证集文件
    train_pairs = [p for p in full_train_ds.pairs if p[0] is not None and p[0].name not in val_names]
    full_train_ds.pairs = train_pairs

    print(f"[Dataset] 训练集: {len(full_train_ds)} (排除验证集后)")
    print(f"[Dataset] 验证集: {len(val_ds)}")
    
    train_loader = DataLoader(
        full_train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    
    # ====== 模型 ======
    model = build_pix2pix_model(
        input_channels=3, cond_channels=3, output_channels=3, base_filters=64,
    ).to(device)
    
    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    image_criterion = PyramidL1SSIMLoss(
        pyramid_weight=100.0,
        ssim_weight=50.0,
        levels=args.pyramid_levels,
    ).to(device)
    gan_criterion = GANLoss(loss_type='lsgan').to(device)
    
    timestamp = int(time.time())
    out_dir = Path(f"./checkpoints/quick_v6_{args.marker}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_ssim = -1.0
    best_epoch = -1
    
    for epoch in range(args.epochs):
        # ====== 训练 ======
        model.train()
        t0 = time.time()
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        n_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            dapi = batch["dapi"].to(device, non_blocking=True)
            real_img = batch["ihc"].to(device, non_blocking=True)
            
            # Discriminator
            optimizer_d.zero_grad(set_to_none=True)
            fake_img = model.generator(dapi, dapi)
            real_pred = model.discriminator(real_img, dapi)
            fake_pred = model.discriminator(fake_img.detach(), dapi)
            loss_d = (gan_criterion(real_pred, True) + gan_criterion(fake_pred, False)) * 0.5
            loss_d.backward()
            optimizer_d.step()
            
            # Generator
            optimizer_g.zero_grad(set_to_none=True)
            fake_img = model.generator(dapi, dapi)
            fake_pred = model.discriminator(fake_img, dapi)
            loss_g_gan = gan_criterion(fake_pred, True)
            loss_g_img = image_criterion(fake_img, real_img)
            loss_g = loss_g_gan + loss_g_img
            loss_g.backward()
            optimizer_g.step()
            
            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()
            n_batches += 1
            
            if (batch_idx + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                avg_loss_g = epoch_loss_g / n_batches
                avg_loss_d = epoch_loss_d / n_batches
                print(f"[Epoch {epoch+1}/{args.epochs}] "
                      f"batch {batch_idx+1}/{len(train_loader)} "
                      f"loss_G={avg_loss_g:.4f} loss_D={avg_loss_d:.4f} "
                      f"time={elapsed:.1f}s")
        
        train_time = time.time() - t0
        avg_loss_g = epoch_loss_g / max(n_batches, 1)
        avg_loss_d = epoch_loss_d / max(n_batches, 1)
        
        print(f"\n[Epoch {epoch+1}] 训练完成: loss_G={avg_loss_g:.4f} loss_D={avg_loss_d:.4f} 时间={train_time:.1f}s")
        
        # ====== 验证 ======
        if (epoch + 1) % args.val_every == 0:
            model.eval()
            val_loss = 0.0
            aggregator = MetricAggregator()
            val_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    dapi = batch["dapi"].to(device)
                    real_img = batch["ihc"].to(device)
                    
                    fake_img = model.generator(dapi, dapi)
                    loss = image_criterion(fake_img, real_img)
                    val_loss += loss.item()
                    aggregator.update(fake_img, real_img)
                    val_count += 1
            
            avg_val_loss = val_loss / max(val_count, 1)
            metrics = aggregator.result()
            avg_val_ssim = metrics["ssim"]
            avg_val_psnr = metrics["psnr"]
            
            print(f"[Val {epoch+1}] loss={avg_val_loss:.4f} SSIM={avg_val_ssim:.4f} PSNR={avg_val_psnr:.2f}")
            
            # 保存最好的模型
            if avg_val_ssim > best_val_ssim:
                best_val_ssim = avg_val_ssim
                best_epoch = epoch + 1
                best_path = out_dir / "best.pt"
                torch.save({
                    "model": model.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "epoch": epoch,
                    "val_ssim": avg_val_ssim,
                    "val_loss": avg_val_loss,
                }, best_path)
                print(f"[Best] epoch={best_epoch} SSIM={best_val_ssim:.4f}")
        
        # 每个 epoch 保存一次
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
        }, out_dir / f"epoch_{epoch+1}.pt")
    
    print(f"\n{'='*60}")
    print(f"[完成] marker={args.marker}")
    print(f"[完成] 最佳验证 SSIM: {best_val_ssim:.4f} (epoch {best_epoch})")
    print(f"[完成] 模型保存位置: {out_dir}")
    print(f"{'='*60}")


def compute_ssim(img1, img2, window_size=11):
    """正确的 SSIM 计算（全局统计版本，与 v6 训练一致）"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 对每个样本独立计算
    B = img1.shape[0]
    ssim_total = 0.0
    
    for b in range(B):
        x = img1[b:b+1]
        y = img2[b:b+1]
        
        mu1 = x.mean(dim=[2, 3], keepdim=True)
        mu2 = y.mean(dim=[2, 3], keepdim=True)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = ((x - mu1) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma2_sq = ((y - mu2) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma12 = ((x - mu1) * (y - mu2)).mean(dim=[2, 3], keepdim=True)
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        ssim_total += ssim_map.mean().item()
    
    return ssim_total / B


if __name__ == "__main__":
    main()
