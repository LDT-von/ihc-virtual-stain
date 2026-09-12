"""改进的训练脚本 v7 - 解决欠拟合问题

关键改进：
1. 训练时使用与最终评估相同的验证集（真实验证集）
2. 增加训练 epoch（15 epoch vs 之前的 8 epoch）
3. 调整损失权重（增大 L1 权重到 150）
4. 减少 Discriminator 训练频率（每 3 个 batch 训一次 D）
5. 学习率余弦衰减
6. 增强数据增强（更激进的亮度/对比度）
"""
import argparse
import sys
import builtins
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random

# 强制 unbuffered output
_orig_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    try:
        _orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        safe = [str(a).encode('gbk', errors='replace').decode('gbk') for a in args]
        _orig_print(*safe, **kwargs)
builtins.print = _flush_print

from src.data.dataset import DAPItoIHCDataset
from src.models.pix2pix_gan import build_pix2pix_model
from src.models.losses import PyramidL1SSIMLoss, GANLoss
from src.metrics.ssim_psnr import MetricAggregator


def parse_args():
    p = argparse.ArgumentParser(description="改进训练 v7")
    p.add_argument("--data_root", type=str,
                   default="E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）")
    p.add_argument("--marker", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_pyr", type=float, default=200.0, help="多尺度 L1 权重")
    p.add_argument("--lambda_ssim", type=float, default=80.0, help="SSIM 权重")
    p.add_argument("--pyramid_levels", type=int, default=4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--d_every_n", type=int, default=3)
    p.add_argument("--resume_ckpt", type=str, default=None)
    p.add_argument("--brightness", type=float, default=0.25)
    p.add_argument("--contrast", type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=" * 60)
    print(f"[v7 Train] marker={args.marker}")
    print(f"[v7] epochs={args.epochs} bs={args.batch_size} lr={args.lr}")
    print(f"[v7] lambda_pyr={args.lambda_pyr} lambda_ssim={args.lambda_ssim}")
    print(f"[v7] d_every_n={args.d_every_n} (D training frequency)")
    print(f"[v7] device={device}")
    print(f"=" * 60)
    
    # ====== 加载训练集（使用完整的 train 目录）=======
    train_ds = DAPItoIHCDataset(
        root=args.data_root,
        marker=args.marker,
        split="train",
        patch_size=256,
        augment=True,
    )
    print(f"[Dataset] 训练集: {len(train_ds)} samples")
    
    # ====== 加载验证集（使用 split_data.py 划分的 val 目录）=======
    val_ds = DAPItoIHCDataset(
        root=args.data_root,
        marker=args.marker,
        split="val",  # 使用独立的 val 目录（约 20% 数据）
        patch_size=256,
        augment=False,
    )
    
    if len(val_ds) == 0:
        print("[Error] 验证集为空！请先运行 split_data.py 划分数据")
        return
    
    print(f"[Dataset] 验证集: {len(val_ds)} samples (20%% 划分)")
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    
    # ====== 模型 ======
    model = build_pix2pix_model(
        input_channels=3, cond_channels=3, output_channels=3, base_filters=64,
    ).to(device)
    print(f"[Model] G params: {sum(p.numel() for p in model.generator.parameters()):,}")
    
    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))
    
    # 学习率衰减
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=args.epochs, eta_min=1e-5)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=args.epochs, eta_min=1e-5)
    
    image_criterion = PyramidL1SSIMLoss(
        pyramid_weight=args.lambda_pyr,
        ssim_weight=args.lambda_ssim,
        levels=args.pyramid_levels,
    ).to(device)
    gan_criterion = GANLoss(loss_type='lsgan').to(device)
    
    # 从 checkpoint 恢复
    start_epoch = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer_g" in ckpt:
            optimizer_g.load_state_dict(ckpt["optimizer_g"])
            optimizer_d.load_state_dict(ckpt["optimizer_d"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"[Resume] loaded {args.resume_ckpt}, start_epoch={start_epoch}")
    
    timestamp = int(time.time())
    out_dir = Path(f"./checkpoints/pix2pix_v7_{args.marker}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")
    
    best_val_ssim = -1.0
    best_epoch = -1
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        n_batches = 0
        d_trained = 0
        
        for batch_idx, batch in enumerate(train_loader):
            dapi = batch["dapi"].to(device, non_blocking=True)
            real_img = batch["ihc"].to(device, non_blocking=True)
            
            # Discriminator (减少训练频率)
            if (batch_idx + 1) % args.d_every_n == 0:
                optimizer_d.zero_grad(set_to_none=True)
                fake_img = model.generator(dapi, dapi)
                real_pred = model.discriminator(real_img, dapi)
                loss_d_real = gan_criterion(real_pred, True)
                fake_pred = model.discriminator(fake_img.detach(), dapi)
                loss_d_fake = gan_criterion(fake_pred, False)
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
                optimizer_d.step()
                epoch_loss_d += loss_d.item()
                d_trained += 1
            else:
                loss_d = torch.tensor(0.0)
            
            # Generator (每个 batch 都训练)
            optimizer_g.zero_grad(set_to_none=True)
            fake_img = model.generator(dapi, dapi)
            fake_pred = model.discriminator(fake_img, dapi)
            loss_g_gan = gan_criterion(fake_pred, True)
            loss_g_content = image_criterion(fake_img, real_img)
            loss_g = loss_g_gan + loss_g_content
            
            if not torch.isfinite(loss_g):
                print(f"[Warn] NaN at epoch {epoch} step {batch_idx}, skip")
                continue
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
            optimizer_g.step()
            
            epoch_loss_g += loss_g.item()
            n_batches += 1
            
            if (batch_idx + 1) % args.log_every == 0:
                print(f"[E{epoch} B{batch_idx+1}/{len(train_loader)}] "
                      f"G={loss_g.item():.3f} (gan={loss_g_gan.item():.3f} content={loss_g_content.item():.3f}) "
                      f"D={loss_d.item():.3f} t={time.time()-t0:.0f}s")
        
        # 学习率衰减
        scheduler_g.step()
        scheduler_d.step()
        
        avg_g = epoch_loss_g / max(n_batches, 1)
        avg_d = epoch_loss_d / max(d_trained, 1) if d_trained > 0 else 0
        elapsed = time.time() - t0
        
        # 验证
        model.eval()
        aggregator = MetricAggregator()
        with torch.no_grad():
            for vb in val_loader:
                dapi = vb["dapi"].to(device)
                real_img = vb["ihc"].to(device)
                fake_img = model.generator(dapi, dapi)
                aggregator.update(fake_img, real_img)
        
        metrics = aggregator.result()
        val_ssim = metrics["ssim"]
        val_psnr = metrics["psnr"]
        
        print(f"[Epoch {epoch}] avg_G={avg_g:.3f} avg_D={avg_d:.3f} "
              f"val_ssim={val_ssim:.4f} val_psnr={val_psnr:.2f} "
              f"t={elapsed:.0f}s lr={scheduler_g.get_last_lr()[0]:.2e}")
        
        log_file.write(f'{{"epoch":{epoch},"loss_g":{avg_g:.4f},"loss_d":{avg_d:.4f},'
                       f'"val_ssim":{val_ssim:.4f},"val_psnr":{val_psnr:.2f},"time":{elapsed:.0f},'
                       f'"lr":{scheduler_g.get_last_lr()[0]:.2e}}}\n')
        log_file.flush()
        
        # 保存 checkpoint
        ckpt_path = out_dir / f"epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "val_ssim": val_ssim,
            "val_psnr": val_psnr,
        }, ckpt_path)
        
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            best_epoch = epoch
            best_path = out_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "val_ssim": val_ssim,
                "val_psnr": val_psnr,
            }, best_path)
            print(f"[Best] Saved best.pt (epoch={epoch}, val_ssim={val_ssim:.4f})")
    
    log_file.close()
    print(f"\n[Done] Best epoch={best_epoch} val_ssim={best_val_ssim:.4f}")
    print(f"[Done] Output dir: {out_dir}")


if __name__ == "__main__":
    main()
