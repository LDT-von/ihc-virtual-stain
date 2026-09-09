"""Pix2Pix GAN 训练入口（简化版）

用法：
    python -m src.train_pix2pix_simple --marker HLA-DR
"""
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data.dataset import DAPItoIHCDataset
from .models.pix2pix_gan import build_pix2pix_model
from .models.losses import CombinedLoss, GANLoss


def parse_args():
    p = argparse.ArgumentParser(description="Pix2Pix GAN 训练（简化版）")
    p.add_argument("--data_root", type=str, 
                   default="E:/aic/初赛数据集（包含训练集和测试集输入）",
                   help="数据根目录")
    p.add_argument("--marker", type=str, default="HLA-DR", help="目标标记")
    p.add_argument("--epochs", type=int, default=30, help="训练轮数")
    p.add_argument("--batch_size", type=int, default=8, help="批次大小")
    p.add_argument("--lr", type=float, default=2e-4, help="学习率")
    p.add_argument("--lambda_l1", type=float, default=100.0, help="L1 损失权重")
    p.add_argument("--lambda_ssim", type=float, default=50.0, help="SSIM 损失权重")
    p.add_argument("--save_every", type=int, default=1, help="保存间隔（轮）")
    p.add_argument("--log_every", type=int, default=50, help="日志间隔（步）")
    return p.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] marker={args.marker} device={device}")
    
    # 数据集
    train_ds = DAPItoIHCDataset(
        root=args.data_root,
        marker=args.marker,
        split="train",
        patch_size=256,
        augment=True,
    )
    
    print(f"[Dataset] train: {len(train_ds)} samples")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 避免多进程问题
        drop_last=True,
        pin_memory=True,
    )
    
    # 模型
    model = build_pix2pix_model(
        input_channels=3,
        cond_channels=3,
        output_channels=3,
        base_filters=64,
    ).to(device)
    
    print(f"[Model] Generator parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    
    # 优化器
    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # 损失函数
    image_criterion = CombinedLoss(
        l1_weight=args.lambda_l1,
        ssim_weight=args.lambda_ssim,
    ).to(device)
    
    gan_criterion = GANLoss(loss_type='lsgan').to(device)
    
    # 输出目录
    timestamp = int(time.time())
    out_dir = Path(f"./checkpoints/pix2pix_{args.marker}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        n_batches = 0
        nan_detected = False
        
        for batch_idx, batch in enumerate(train_loader):
            dapi = batch["dapi"].to(device, non_blocking=True)
            real_img = batch["ihc"].to(device, non_blocking=True)
            
            # ========== 训练判别器 ==========
            optimizer_d.zero_grad(set_to_none=True)
            fake_img = model.generator(dapi, dapi)
            
            # 判别器损失
            loss_d = compute_discriminator_loss(model, real_img, fake_img, dapi, gan_criterion)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()
            
            # ========== 训练生成器 ==========
            optimizer_g.zero_grad(set_to_none=True)
            fake_img = model.generator(dapi, dapi)
            
            # 生成器损失
            loss_g, loss_details = compute_generator_loss(
                model, fake_img, real_img, dapi, gan_criterion, image_criterion
            )
            
            # NaN 检测
            if not torch.isfinite(loss_g):
                print(f"[Warn] NaN detected at epoch {epoch}, step {batch_idx}, skipping...")
                nan_detected = True
                break
            
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
            optimizer_g.step()
            
            # 统计
            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()
            n_batches += 1
            global_step += 1
            
            if batch_idx % args.log_every == 0:
                print(f"[Epoch {epoch} Batch {batch_idx}/{len(train_loader)}] "
                      f"loss_G={loss_g.item():.4f} loss_D={loss_d.item():.4f}")
        
        if nan_detected:
            print(f"[Train] NaN detected, stopping training")
            break
        
        avg_loss_g = epoch_loss_g / max(n_batches, 1)
        avg_loss_d = epoch_loss_d / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] loss_G={avg_loss_g:.4f} loss_D={avg_loss_d:.4f} time={elapsed:.1f}s")
        
        # 保存检查点
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = out_dir / f"epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
            }, ckpt_path)
            print(f"[Train] 保存 {ckpt_path}")
    
    # 最终保存
    final_path = out_dir / "final.pt"
    torch.save({
        "epoch": args.epochs - 1,
        "model": model.state_dict(),
    }, final_path)
    print(f"[Train] 训练完成 -> {final_path}")


def compute_discriminator_loss(model, real_img, fake_img, dapi, gan_criterion):
    real_pred = model.discriminator(real_img, dapi)
    loss_real = gan_criterion(real_pred, True)
    fake_pred = model.discriminator(fake_img.detach(), dapi)
    loss_fake = gan_criterion(fake_pred, False)
    return (loss_real + loss_fake) * 0.5


def compute_generator_loss(model, fake_img, real_img, dapi, gan_criterion, image_criterion):
    fake_pred = model.discriminator(fake_img, dapi)
    loss_gan = gan_criterion(fake_pred, True)
    loss_content = image_criterion(fake_img, real_img)
    loss_g = loss_gan + loss_content
    return loss_g, {'gan': loss_gan.item(), 'content': loss_content.item()}


if __name__ == "__main__":
    main()
