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
    p.add_argument("--val_split", type=float, default=0.05, help="验证集比例")
    p.add_argument("--full_data", action="store_true",
                   help="全量训练：不划分验证集（用于在已选好 best.pt 基础上做全量微调）")
    p.add_argument("--lambda_perceptual", type=float, default=10.0, help="感知损失权重（0 表示禁用）")
    p.add_argument("--lambda_edge", type=float, default=5.0, help="边缘损失权重")
    p.add_argument("--lambda_css", type=float, default=20.0, help="CSS Loss 权重（DAPI↔fake↔IHC；0 表示禁用）。论文 CSSP2P GAN 推荐 10-25。")
    p.add_argument("--use_lpips", action="store_true", help="用 LPIPS 替换自实现 VGG Perceptual（与人类感知更对齐）")
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

    # 划分 val（按比例随机抽，取 IHC 配对的子集做 SSIM/PSNR 评估）
    from torch.utils.data import Subset
    import random
    if args.full_data:
        print(f"[Split] full-data 模式：使用全部 {len(train_ds)} 个训练样本，不划分验证集")
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        val_loader = None
    else:
        val_size = max(1, int(len(train_ds) * args.val_split))
        val_size = min(val_size, len(train_ds) - 1)
        rng = random.Random(42)
        val_indices = sorted(rng.sample(range(len(train_ds)), val_size))
        train_indices = [i for i in range(len(train_ds)) if i not in set(val_indices)]
        val_ds = Subset(DAPItoIHCDataset(
            root=args.data_root, marker=args.marker, split="train",
            patch_size=256, augment=False,
        ), val_indices)
        train_ds = Subset(train_ds, train_indices)
        print(f"[Split] train={len(train_ds)}  val={len(val_ds)}")

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
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
        edge_weight=args.lambda_edge,
        css_weight=args.lambda_css,    # ← NEW
    ).to(device)

    # 可选感知损失（LPIPS 优先；否则沿用旧 VGG Perceptual）
    perceptual_criterion = None
    if args.lambda_perceptual > 0:
        if args.use_lpips:
            from .models.losses import LPIPSLoss
            perceptual_criterion = LPIPSLoss(net='vgg').to(device)
        else:
            from .models.losses import PerceptualLoss
            perceptual_criterion = PerceptualLoss().to(device)
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

    best_ssim = -1.0
    best_epoch = -1

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
            loss_g_content = image_criterion(fake_img, real_img, dapi=dapi)   # ← dapi 传入
            if perceptual_criterion is not None:
                loss_g_content = loss_g_content + args.lambda_perceptual * perceptual_criterion(fake_img, real_img)
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

        # val 评估，找最优（full-data 微调模式下跳过：样本已被训练过，不能再用作选优依据）
        if val_loader is not None:
            model.eval()
            from .metrics.ssim_psnr import MetricAggregator
            val_metric = MetricAggregator()
            with torch.inference_mode():
                for vb in val_loader:
                    v_dapi = vb["dapi"].to(device, non_blocking=True)
                    v_real = vb["ihc"].to(device, non_blocking=True)
                    v_fake = model.generator(v_dapi, v_dapi)
                    val_metric.update(v_fake, v_real)
            vm = val_metric.result()
            print(f"[Val {epoch}] SSIM={vm['ssim']:.4f} PSNR={vm['psnr']:.2f}")
            log_file.write(json.dumps({"epoch": epoch, "val_ssim": vm["ssim"], "val_psnr": vm["psnr"]}) + "\n")
            log_file.flush()
            if vm["ssim"] > best_ssim:
                best_ssim = vm["ssim"]
                best_epoch = epoch
                torch.save({"epoch": epoch, "model": model.state_dict(), "val_ssim": vm["ssim"]}, out_dir / "best.pt")
                print(f"[Best] new best SSIM={vm['ssim']:.4f} @ epoch {epoch}")

    final_path = out_dir / "final.pt"
    torch.save({"epoch": start_epoch + args.epochs - 1, "model": model.state_dict()}, final_path)
    print(f"[Train] done -> {final_path}")
    if best_epoch >= 0:
        print(f"[Train] best.pt SSIM={best_ssim:.4f} @ epoch {best_epoch}")
    else:
        print("[Train] full-data 模式：未划分验证集，本次仅保存 final.pt")
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
