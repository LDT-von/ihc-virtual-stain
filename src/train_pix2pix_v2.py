"""Pix2Pix GAN 训练脚本（强化版）

用法：
    # 从头训练
    python -m src.train_pix2pix_v2 --marker HLA-DR --epochs 30 --batch_size 16 \
        --resume_ckpt checkpoints/pix2pix_HLA-DR_xxx/epoch14.pt

    # 新 marker
    python -m src.train_pix2pix_v2 --marker CD68 --epochs 30 --batch_size 16

改进：
- 支持从已有 checkpoint 恢复（含优化器状态）
- L1 + SSIM 损失权重可调
- 训练日志写入文件
- 每个 epoch 末保存 ckpt
"""
import argparse
import json
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
    p = argparse.ArgumentParser(description="Pix2Pix GAN 训练 v2")
    p.add_argument("--data_root", type=str,
                   default="E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）",
                   help="数据根目录")
    p.add_argument("--marker", type=str, default="HLA-DR", help="目标标记")
    p.add_argument("--epochs", type=int, default=30, help="训练轮数")
    p.add_argument("--batch_size", type=int, default=16, help="批次大小")
    p.add_argument("--lr", type=float, default=2e-4, help="学习率")
    p.add_argument("--lambda_l1", type=float, default=100.0, help="L1 损失权重")
    p.add_argument("--lambda_ssim", type=float, default=50.0, help="SSIM 损失权重")
    p.add_argument("--save_every", type=int, default=1, help="保存间隔（轮）")
    p.add_argument("--log_every", type=int, default=50, help="日志间隔（步）")
    p.add_argument("--resume_ckpt", type=str, default=None, help="从已有 .pt 恢复")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader 进程数")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] marker={args.marker} device={device} epochs={args.epochs} bs={args.batch_size}")

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
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    model = build_pix2pix_model(
        input_channels=3,
        cond_channels=3,
        output_channels=3,
        base_filters=64,
    ).to(device)

    print(f"[Model] G params: {sum(p.numel() for p in model.generator.parameters()):,}")

    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    image_criterion = CombinedLoss(
        l1_weight=args.lambda_l1,
        ssim_weight=args.lambda_ssim,
    ).to(device)
    gan_criterion = GANLoss(loss_type='lsgan').to(device)

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
    out_dir = Path(f"./checkpoints/pix2pix_v2_{args.marker}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    log_file = open(log_path, "a", encoding="utf-8")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        n_batches = 0
        nan_detected = False

        for batch_idx, batch in enumerate(train_loader):
            dapi = batch["dapi"].to(device, non_blocking=True)
            real_img = batch["ihc"].to(device, non_blocking=True)

            # D
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

            # G
            optimizer_g.zero_grad(set_to_none=True)
            fake_img = model.generator(dapi, dapi)
            fake_pred = model.discriminator(fake_img, dapi)
            loss_g_gan = gan_criterion(fake_pred, True)
            loss_g_content = image_criterion(fake_img, real_img)
            loss_g = loss_g_gan + loss_g_content

            if not torch.isfinite(loss_g):
                print(f"[Warn] NaN at epoch {epoch} step {batch_idx}, skip")
                nan_detected = True
                break

            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
            optimizer_g.step()

            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()
            n_batches += 1

            if batch_idx % args.log_every == 0:
                print(f"[E{epoch} B{batch_idx}/{len(train_loader)}] "
                      f"G={loss_g.item():.3f} D={loss_d.item():.3f} "
                      f"gan={loss_g_gan.item():.3f} content={loss_g_content.item():.3f}")

        if nan_detected:
            break

        avg_g = epoch_loss_g / max(n_batches, 1)
        avg_d = epoch_loss_d / max(n_batches, 1)
        elapsed = time.time() - t0
        log_entry = {
            "epoch": epoch,
            "loss_g": avg_g,
            "loss_d": avg_d,
            "elapsed": elapsed,
        }
        print(f"[Epoch {epoch}] G={avg_g:.4f} D={avg_d:.4f} time={elapsed:.1f}s")
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = out_dir / f"epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
            }, ckpt_path)

    final_path = out_dir / "final.pt"
    torch.save({"epoch": start_epoch + args.epochs - 1, "model": model.state_dict()}, final_path)
    print(f"[Train] done -> {final_path}")
    log_file.close()


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
