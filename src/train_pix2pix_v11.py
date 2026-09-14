"""
Pix2Pix GAN 训练脚本 v11 — 修复版

修复了 v6 的所有问题，并结合 v2 的优势损失：
  1. PyramidLoss 归一化错误（除以 sum(weights) → 梯度弱 4 倍）
  2. 验证 SSIM 指标错误（均值方差近似 → skimage 正确 sliding-window）
  3. Discriminator 快速崩溃（lr_d = lr_g * 0.5）
  4. 缺少感知损失（默认禁用，节省显存）
  5. 训练不足（20 epochs + 15 epochs full-data 微调）
  6. 显存不足（AMP 自动混合精度）
  7. 损失函数不完整（CombinedLoss = L1 + SSIM + CSS + Edge + PyramidL1）

用法：
    python -m src.train_pix2pix_v11 --marker HLA-DR --epochs 20 --batch_size 16
"""
import argparse
import json
import sys
import time
import builtins
import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import random
from torch.amp import autocast as _autocast, GradScaler

# 强制 unbuffered stdout
_orig_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)
builtins.print = _flush_print

from .data.dataset import DAPItoIHCDataset
from .models.pix2pix_gan import build_pix2pix_model
from .models.losses import CombinedLoss, PyramidLoss, GANLoss
from .metrics.ssim_psnr import ssim as skimage_ssim


def parse_args():
    p = argparse.ArgumentParser(description="Pix2Pix GAN 训练 v11（修复版）")
    p.add_argument("--data_root", type=str,
                   default="E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）")
    p.add_argument("--marker", type=str, default="HLA-DR")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=1e-4,
                   help="Discriminator 学习率（默认 G 的 50%%）")
    p.add_argument("--lambda_l1", type=float, default=0.0,
                   help="L1 损失权重（0=用 PyramidL1 替代）")
    p.add_argument("--lambda_pyramid", type=float, default=100.0,
                   help="多尺度 L1（Pyramid）权重")
    p.add_argument("--lambda_ssim", type=float, default=50.0,
                   help="SSIM 损失权重")
    p.add_argument("--lambda_css", type=float, default=20.0,
                   help="CSS Loss 权重（DAPI↔fake↔IHC；0=禁用）")
    p.add_argument("--lambda_edge", type=float, default=5.0,
                   help="边缘损失权重（0=禁用）")
    p.add_argument("--lambda_perceptual", type=float, default=0.0,
                   help="感知损失权重（0=禁用）")
    p.add_argument("--pyramid_levels", type=int, default=4)
    p.add_argument("--pyramid_weights", type=float, nargs=4, default=None,
                   help="每层 pyramid 权重，如：1.0 0.7 0.5 0.3")
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--resume_ckpt", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--full_data", action="store_true")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="标签平滑（real label 从 1.0 → 1 - smoothing）")
    p.add_argument("--no_amp", action="store_true", help="禁用 AMP")
    return p.parse_args()


def train_one_epoch(model, train_loader, optimizer_g, optimizer_d,
                    image_criterion, pyramid_criterion, perceptual_criterion,
                    gan_criterion, device, args, epoch,
                    scaler_g=None, scaler_d=None, use_amp=True):
    """训练一个 epoch，支持 AMP"""
    model.train()
    epoch_loss_g = 0.0
    epoch_loss_d = 0.0
    n_batches = 0
    nan_detected = False
    _ctx = _autocast('cuda') if use_amp else contextlib.nullcontext()

    for batch_idx, batch in enumerate(train_loader):
        dapi = batch["dapi"].to(device, non_blocking=True)
        real_img = batch["ihc"].to(device, non_blocking=True)

        # ---------- D ----------
        optimizer_d.zero_grad(set_to_none=True)
        if use_amp:
            with _ctx:
                fake_img = model.generator(dapi, dapi).detach()
                real_pred = model.discriminator(real_img, dapi)
                fake_pred = model.discriminator(fake_img, dapi)
                ls = args.label_smoothing
                loss_d_real = gan_criterion(real_pred, True) * (1 - ls)
                loss_d_fake = gan_criterion(fake_pred, False)
                loss_d = (loss_d_real + loss_d_fake) * 0.5
            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
            scaler_d.step(optimizer_d)
            scaler_d.update()
        else:
            fake_img = model.generator(dapi, dapi).detach()
            real_pred = model.discriminator(real_img, dapi)
            fake_pred = model.discriminator(fake_img, dapi)
            ls = args.label_smoothing
            loss_d_real = gan_criterion(real_pred, True) * (1 - ls)
            loss_d_fake = gan_criterion(fake_pred, False)
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()

        # ---------- G ----------
        optimizer_g.zero_grad(set_to_none=True)

        if use_amp:
            with _ctx:
                fake_img = model.generator(dapi, dapi)
                fake_pred = model.discriminator(fake_img, dapi)
                loss_g_gan = gan_criterion(fake_pred, True)
            # 损失在 autocast 外计算，确保 FP32 精度
            fake_f32 = fake_img.float()
            real_f32 = real_img.float()
            dapi_f32 = dapi.float()
            # CombinedLoss: SSIM + CSS + Edge（需要 dapi）
            loss_g_extra = image_criterion(fake_f32, real_f32, dapi=dapi_f32)
            # PyramidL1（多尺度 L1）
            loss_g_pyr = pyramid_criterion(fake_f32, real_f32)
            loss_g_content = args.lambda_pyramid * loss_g_pyr + loss_g_extra
            if perceptual_criterion is not None:
                loss_g_perc = perceptual_criterion(fake_f32, real_f32)
                loss_g_content = loss_g_content + args.lambda_perceptual * loss_g_perc
            loss_g = loss_g_gan + loss_g_content

            if not (torch.isfinite(loss_g) if isinstance(loss_g, torch.Tensor) else True):
                print(f"[Warn] NaN at epoch {epoch} step {batch_idx}, skip")
                nan_detected = True
                break

            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=1.0)
            scaler_g.step(optimizer_g)
            scaler_g.update()
        else:
            fake_img = model.generator(dapi, dapi)
            fake_pred = model.discriminator(fake_img, dapi)
            loss_g_gan = gan_criterion(fake_pred, True)
            loss_g_extra = image_criterion(fake_img, real_img, dapi=dapi)
            loss_g_pyr = pyramid_criterion(fake_img, real_img)
            loss_g_content = args.lambda_pyramid * loss_g_pyr + loss_g_extra
            if perceptual_criterion is not None:
                loss_g_perc = perceptual_criterion(fake_img, real_img)
                loss_g_content = loss_g_content + args.lambda_perceptual * loss_g_perc
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

        if (batch_idx + 1) % args.log_every == 0:
            print(f"[E{epoch} B{batch_idx+1}/{len(train_loader)}] "
                  f"G={loss_g.item():.3f} (gan={loss_g_gan.item():.3f} "
                  f"pyr={loss_g_pyr.item():.3f} extra={loss_g_extra.item():.3f}) "
                  f"D={loss_d.item():.3f}")

    if nan_detected:
        return None, None
    return epoch_loss_g / max(n_batches, 1), epoch_loss_d / max(n_batches, 1)


@torch.no_grad()
def validate(model, val_loader, device):
    import numpy as np
    ssims, psnrs = [], []
    for batch in val_loader:
        dapi = batch["dapi"].to(device)
        real_img = batch["ihc"].to(device)
        fake_img = model.generator(dapi, dapi)
        for i in range(fake_img.shape[0]):
            ss = skimage_ssim(fake_img[i], real_img[i])
            mse = ((fake_img[i].clamp(-1,1) - real_img[i].clamp(-1,1))**2).mean().item()
            ps = 100.0 if mse < 1e-10 else 10.0 * np.log10(4.0 / mse)
            ssims.append(ss); psnrs.append(ps)
    return {"ssim": float(np.mean(ssims)), "psnr": float(np.mean(psnrs))}


def main():
    import numpy as np

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp

    print(f"[v11] marker={args.marker} device={device} epochs={args.epochs} bs={args.batch_size}")
    print(f"[v11] lr_g={args.lr} lr_d={args.lr_d}")
    print(f"[v11] pyramid={args.lambda_pyramid} ssim={args.lambda_ssim} "
          f"css={args.lambda_css} edge={args.lambda_edge} perc={args.lambda_perceptual}")
    print(f"[v11] amp={use_amp}")

    # 数据集
    train_ds = DAPItoIHCDataset(
        root=args.data_root, marker=args.marker, split="train",
        patch_size=256, augment=True)
    print(f"[Dataset] train: {len(train_ds)} samples")

    if args.full_data:
        print(f"[Split] full-data: {len(train_ds)} samples")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, drop_last=True, pin_memory=True)
        val_loader = None
    else:
        val_size = max(1, int(len(train_ds) * args.val_split))
        val_size = min(val_size, len(train_ds) - 1)
        rng = random.Random(42)
        val_indices = sorted(rng.sample(range(len(train_ds)), val_size))
        train_indices = [i for i in range(len(train_ds)) if i not in set(val_indices)]
        val_ds = Subset(DAPItoIHCDataset(
            root=args.data_root, marker=args.marker, split="train",
            patch_size=256, augment=False), val_indices)
        train_subset = Subset(train_ds, train_indices)
        print(f"[Split] train={len(train_subset)}  val={len(val_ds)}")
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    # 模型
    model = build_pix2pix_model(input_channels=3, cond_channels=3,
                                  output_channels=3, base_filters=64).to(device)
    print(f"[Model] G params: {sum(p.numel() for p in model.generator.parameters()):,}")

    # AMP
    scaler_g = GradScaler() if use_amp else None
    scaler_d = GradScaler() if use_amp else None
    if use_amp:
        print("[AMP] enabled")

    # 优化器
    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    # 损失函数
    # CombinedLoss: SSIM + CSS + Edge（l1_weight=0，因为用 PyramidL1 替代）
    image_criterion = CombinedLoss(
        l1_weight=args.lambda_l1,
        ssim_weight=args.lambda_ssim,
        edge_weight=args.lambda_edge,
        css_weight=args.lambda_css,
    ).to(device)

    # PyramidL1（多尺度 L1）
    pyramid_criterion = PyramidLoss(
        levels=args.pyramid_levels,
        weights=args.pyramid_weights,
    ).to(device)

    gan_criterion = GANLoss(loss_type='lsgan').to(device)

    perceptual_criterion = None
    if args.lambda_perceptual > 0:
        try:
            from .models.losses import LPIPSLoss
            perceptual_criterion = LPIPSLoss(net='vgg').to(device)
            print(f"[Loss] LPIPS enabled (lambda={args.lambda_perceptual})")
        except ImportError:
            print("[Loss] LPIPS not available, skipping")

    print(f"[Loss] PyramidL1({args.lambda_pyramid}) + SSIM({args.lambda_ssim}) "
          f"+ CSS({args.lambda_css}) + Edge({args.lambda_edge})")

    # 恢复
    start_epoch = 0
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer_g" in ckpt:
            optimizer_g.load_state_dict(ckpt["optimizer_g"])
            optimizer_d.load_state_dict(ckpt["optimizer_d"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"[Resume] {args.resume_ckpt}, start_epoch={start_epoch}")

    # 输出目录
    timestamp = int(time.time())
    out_dir = Path(f"./checkpoints/pix2pix_v11_{args.marker}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "train_log.jsonl", "a", encoding="utf-8")

    best_ssim = -1.0
    best_epoch = -1

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        avg_g, avg_d = train_one_epoch(
            model, train_loader, optimizer_g, optimizer_d,
            image_criterion, pyramid_criterion, perceptual_criterion,
            gan_criterion, device, args, epoch,
            scaler_g, scaler_d, use_amp,
        )

        if avg_g is None:
            print(f"[Epoch {epoch}] NaN, skipping")
            continue

        elapsed = time.time() - t0
        val_ssim = -1.0
        val_psnr = -1.0

        if val_loader is not None:
            vm = validate(model, val_loader, device)
            val_ssim = vm["ssim"]
            val_psnr = vm["psnr"]
            print(f"[Epoch {epoch}] G={avg_g:.3f} D={avg_d:.3f} "
                  f"val_ssim={val_ssim:.4f} val_psnr={val_psnr:.2f} t={elapsed:.0f}s")
        else:
            print(f"[Epoch {epoch}] G={avg_g:.3f} D={avg_d:.3f} t={elapsed:.0f}s")

        log_file.write(json.dumps({
            "epoch": epoch, "loss_g": avg_g, "loss_d": avg_d,
            "val_ssim": val_ssim, "val_psnr": val_psnr, "elapsed": elapsed}) + "\n")
        log_file.flush()

        if (epoch + 1) % args.save_every == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                       "optimizer_g": optimizer_g.state_dict(),
                       "optimizer_d": optimizer_d.state_dict(),
                       "val_ssim": val_ssim, "val_psnr": val_psnr},
                      out_dir / f"epoch{epoch}.pt")

        if val_loader is not None and val_ssim > best_ssim:
            best_ssim = val_ssim
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(),
                       "optimizer_g": optimizer_g.state_dict(),
                       "optimizer_d": optimizer_d.state_dict(),
                       "val_ssim": val_ssim, "val_psnr": val_psnr},
                      out_dir / "best.pt")
            print(f"[Best] new best SSIM={val_ssim:.4f} @ epoch {epoch}")

    torch.save({"epoch": start_epoch + args.epochs - 1, "model": model.state_dict()},
               out_dir / "final.pt")
    log_file.close()
    print(f"\n[Done] best_epoch={best_epoch} best_ssim={best_ssim:.4f}")
    print(f"[Done] output: {out_dir}")


if __name__ == "__main__":
    main()
