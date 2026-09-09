"""Pix2Pix GAN 训练入口

用法：
    python -m src.train_pix2pix --marker HLA-DR
    python -m src.train_pix2pix --marker HLA-DR --epochs 100 --batch_size 8

特点：
- L1 Loss + SSIM Loss + Perceptual Loss 组合
- GAN 对抗训练
- 支持 AMP 混合精度训练
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
    p = argparse.ArgumentParser(description="Pix2Pix GAN 训练")
    p.add_argument("--data_root", type=str, 
                   default="E:/aic/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）",
                   help="数据根目录")
    p.add_argument("--marker", type=str, default="HLA-DR", help="目标标记")
    p.add_argument("--epochs", type=int, default=100, help="训练轮数")
    p.add_argument("--batch_size", type=int, default=8, help="批次大小")
    p.add_argument("--lr", type=float, default=2e-4, help="学习率")
    p.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")
    p.add_argument("--lambda_l1", type=float, default=100.0, help="L1 损失权重")
    p.add_argument("--lambda_ssim", type=float, default=50.0, help="SSIM 损失权重")
    p.add_argument("--gan_mode", type=str, default="lsgan", choices=["vanilla", "lsgan"], help="GAN 模式")
    p.add_argument("--base_filters", type=int, default=64, help="基础滤波器数量")
    p.add_argument("--patch_size", type=int, default=256, help="Patch 尺寸")
    p.add_argument("--device", type=str, default="cuda", help="设备")
    p.add_argument("--resume", type=str, default=None, help="恢复训练路径")
    p.add_argument("--save_every", type=int, default=10, help="保存间隔（轮）")
    p.add_argument("--log_every", type=int, default=50, help="日志间隔（步）")
    p.add_argument("--amp", action="store_true", help="启用 AMP 混合精度")
    return p.parse_args()


def compute_discriminator_loss(
    model: nn.Module,
    real_img: torch.Tensor,
    fake_img: torch.Tensor,
    dapi: torch.Tensor,
    gan_criterion: nn.Module,
    device: torch.Tensor.device,
) -> torch.Tensor:
    """计算判别器损失"""
    # 真实图像的判别损失
    real_pred = model.discriminator(real_img, dapi)
    loss_real = gan_criterion(real_pred, True)
    
    # 生成图像的判别损失
    fake_pred = model.discriminator(fake_img.detach(), dapi)
    loss_fake = gan_criterion(fake_pred, False)
    
    return (loss_real + loss_fake) * 0.5


def compute_generator_loss(
    model: nn.Module,
    fake_img: torch.Tensor,
    real_img: torch.Tensor,
    dapi: torch.Tensor,
    gan_criterion: nn.Module,
    image_criterion: nn.Module,
    lambda_l1: float,
    lambda_ssim: float,
) -> tuple[torch.Tensor, dict]:
    """计算生成器损失"""
    # GAN 损失：让判别器认为生成图像是真实的
    fake_pred = model.discriminator(fake_img, dapi)
    loss_gan = gan_criterion(fake_pred, True)
    
    # L1 + SSIM 损失
    loss_content = image_criterion(fake_img, real_img)
    
    # 总损失
    loss_g = loss_gan + loss_content
    
    return loss_g, {
        'gan': loss_gan.item(),
        'content': loss_content.item(),
    }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer_g: optim.Optimizer,
    optimizer_d: optim.Optimizer,
    image_criterion: nn.Module,
    gan_criterion: nn.Module,
    device: torch.Tensor.device,
    epoch: int,
    lambda_l1: float,
    lambda_ssim: float,
    log_every: int,
    scaler: torch.cuda.amp.GradScaler = None,
) -> dict:
    """训练一个 epoch"""
    model.train()
    
    total_loss_g = 0.0
    total_loss_d = 0.0
    total_gan = 0.0
    total_content = 0.0
    n_batches = 0
    global_step = epoch * len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        dapi = batch["dapi"].to(device, non_blocking=True)
        real_img = batch["ihc"].to(device, non_blocking=True)
        
        # ========== 训练判别器 ==========
        optimizer_d.zero_grad(set_to_none=True)
        
        # 生成假图像
        fake_img = model.generator(dapi, dapi)
        
        # 判别器损失
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss_d = compute_discriminator_loss(
                    model, real_img, fake_img, dapi, gan_criterion, device
                )
            scaler.scale(loss_d).backward()
            scaler.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
            scaler.step(optimizer_d)
            scaler.update()  # 每次更新后必须调用 update()
        else:
            loss_d = compute_discriminator_loss(
                model, real_img, fake_img, dapi, gan_criterion, device
            )
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()
        
        # ========== 训练生成器 ==========
        optimizer_g.zero_grad(set_to_none=True)
        
        # 重新生成假图像（确保与判别器看到的一致）
        fake_img = model.generator(dapi, dapi)
        
        # 生成器损失
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss_g, loss_details = compute_generator_loss(
                    model, fake_img, real_img, dapi, gan_criterion, image_criterion,
                    lambda_l1, lambda_ssim
                )
            scaler.scale(loss_g).backward()
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
            scaler.step(optimizer_g)
            scaler.update()  # 每次更新后必须调用 update()
        else:
            loss_g, loss_details = compute_generator_loss(
                model, fake_img, real_img, dapi, gan_criterion, image_criterion,
                lambda_l1, lambda_ssim
            )
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
            optimizer_g.step()
        
        # 统计
        total_loss_g += loss_g.item()
        total_loss_d += loss_d.item()
        total_gan += loss_details['gan']
        total_content += loss_details['content']
        n_batches += 1
        global_step += 1
        
        # 日志
        if batch_idx % log_every == 0:
            print(f"[Epoch {epoch} Batch {batch_idx}/{len(train_loader)}] "
                  f"loss_G={loss_g.item():.4f} loss_D={loss_d.item():.4f} "
                  f"gan={loss_details['gan']:.4f} content={loss_details['content']:.4f}")
    
    return {
        'loss_g': total_loss_g / max(n_batches, 1),
        'loss_d': total_loss_d / max(n_batches, 1),
        'gan': total_gan / max(n_batches, 1),
        'content': total_content / max(n_batches, 1),
    }


def validate(model: nn.Module, val_loader: DataLoader, image_criterion: nn.Module, device: torch.Tensor.device) -> dict:
    """验证"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            dapi = batch["dapi"].to(device, non_blocking=True)
            real_img = batch["ihc"].to(device, non_blocking=True)
            
            fake_img = model.generator(dapi, dapi)
            loss = image_criterion(fake_img, real_img)
            total_loss += loss.item()
            n_batches += 1
    
    return {'val_loss': total_loss / max(n_batches, 1)}


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Train] marker={args.marker} device={device}")
    
    # 数据集
    train_ds = DAPItoIHCDataset(
        root=args.data_root,
        marker=args.marker,
        split="train",
        patch_size=args.patch_size,
        augment=True,
    )
    val_ds = DAPItoIHCDataset(
        root=args.data_root,
        marker=args.marker,
        split="val",
        patch_size=args.patch_size,
        augment=False,
    )
    
    print(f"[Dataset] train: {len(train_ds)} samples, val: {len(val_ds)} samples")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    ) if len(val_ds) > 0 else None
    
    # 模型
    model = build_pix2pix_model(
        input_channels=3,
        cond_channels=3,
        output_channels=3,
        base_filters=args.base_filters,
    ).to(device)
    
    print(f"[Model] Generator parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"[Model] Discriminator parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")
    
    # 优化器
    optimizer_g = optim.Adam(
        model.generator.parameters(),
        lr=args.lr,
        betas=(args.beta1, 0.999),
    )
    optimizer_d = optim.Adam(
        model.discriminator.parameters(),
        lr=args.lr,
        betas=(args.beta1, 0.999),
    )
    
    # 损失函数
    image_criterion = CombinedLoss(
        l1_weight=args.lambda_l1,
        ssim_weight=args.lambda_ssim,
    ).to(device)
    
    gan_criterion = GANLoss(loss_type=args.gan_mode).to(device)
    
    # AMP
    scaler = torch.amp.GradScaler('cuda') if args.amp and device.type == "cuda" else None
    
    # 恢复
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        optimizer_d.load_state_dict(ckpt["optimizer_d"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Train] 恢复自 {args.resume}, epoch={start_epoch}")
    
    # 输出目录
    timestamp = int(time.time())
    out_dir = Path(f"./checkpoints/pix2pix_{args.marker}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        
        # 训练
        train_stats = train_epoch(
            model, train_loader,
            optimizer_g, optimizer_d,
            image_criterion, gan_criterion,
            device, epoch,
            args.lambda_l1, args.lambda_ssim,
            args.log_every,
            scaler,
        )
        
        # 验证
        val_stats = validate(model, val_loader, image_criterion, device) if val_loader else {}
        
        elapsed = time.time() - t0
        
        print(f"[Epoch {epoch}] "
              f"loss_G={train_stats['loss_g']:.4f} loss_D={train_stats['loss_d']:.4f} "
              f"val_loss={val_stats.get('val_loss', 'N/A')} "
              f"time={elapsed:.1f}s")
        
        # 保存检查点
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = out_dir / f"epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "train_stats": train_stats,
                "val_stats": val_stats,
                "args": vars(args),
            }, ckpt_path)
            print(f"[Train] 保存 {ckpt_path}")
        
        # 保存最佳模型
        if val_loader and val_stats.get('val_loss', float('inf')) < best_val_loss:
            best_val_loss = val_stats['val_loss']
            best_path = out_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
            }, best_path)
            print(f"[Train] 保存最佳模型 val_loss={best_val_loss:.4f}")
    
    print(f"[Train] 训练完成 -> {out_dir}")


if __name__ == "__main__":
    main()
