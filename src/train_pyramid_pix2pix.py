"""PyramidPix2Pix 训练脚本 (v6)

基于 train_pix2pix_v2.py，但用 PyramidL1SSIMLoss 替换 CombinedLoss。
架构不变（U-Net + 自注意力），损失函数升级（多尺度 L1 + SSIM）。

参考：
- Liu et al., "BCI: Breast Cancer Immunohistochemical Image Generation
  through Pyramid Pix2pix", CVPR 2022 Workshop
"""
import argparse
import sys
import builtins
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import random

# 保留原始 print，用 unbuffered 包装，强制每次 print 后 flush
_orig_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

builtins.print = _flush_print

from .data.dataset import DAPItoIHCDataset
from .models.pix2pix_gan import build_pix2pix_model
from .models.losses import PyramidL1SSIMLoss, GANLoss, LPIPSLoss, PerceptualLoss


def parse_args():
    p = argparse.ArgumentParser(description="PyramidPix2Pix 训练 v6")
    p.add_argument("--data_root", type=str,
                   default="E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）")
    p.add_argument("--marker", type=str, default="HLA-DR")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_pyr", type=float, default=100.0, help="多尺度 L1 权重")
    p.add_argument("--lambda_ssim", type=float, default=50.0, help="SSIM 权重")
    p.add_argument("--lambda_perceptual", type=float, default=10.0, help="感知损失权重（0 禁用）")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--resume_ckpt", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--full_data", action="store_true",
                   help="全量训练：不划分验证集")
    p.add_argument("--pyramid_levels", type=int, default=4)
    p.add_argument("--use_lpips", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v6 Train] marker={args.marker} device={device} epochs={args.epochs} bs={args.batch_size}")
    print(f"[v6] PyramidLevels={args.pyramid_levels} lambda_pyr={args.lambda_pyr} lambda_ssim={args.lambda_ssim}")

    train_ds = DAPItoIHCDataset(
        root=args.data_root,
        marker=args.marker,
        split="train",
        patch_size=256,
        augment=True,
    )
    print(f"[Dataset] train: {len(train_ds)} samples")

    if args.full_data:
        print(f"[Split] full-data mode: using all {len(train_ds)} samples")
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True, pin_memory=True,
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
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    model = build_pix2pix_model(
        input_channels=3, cond_channels=3, output_channels=3, base_filters=64,
    ).to(device)
    print(f"[Model] G params: {sum(p.numel() for p in model.generator.parameters()):,}")

    optimizer_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # v6 核心：PyramidL1SSIMLoss 替换 CombinedLoss
    image_criterion = PyramidL1SSIMLoss(
        pyramid_weight=args.lambda_pyr,
        ssim_weight=args.lambda_ssim,
        levels=args.pyramid_levels,
    ).to(device)
    print(f"[Loss] PyramidL1SSIMLoss (L1 pyramid + SSIM)")

    perceptual_criterion = None
    if args.lambda_perceptual > 0:
        if args.use_lpips:
            perceptual_criterion = LPIPSLoss(net='vgg').to(device)
        else:
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
    out_dir = Path(f"./checkpoints/pix2pix_v6_{args.marker}_{timestamp}")
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
            loss_g_content = image_criterion(fake_img, real_img)  # ← PyramidLoss, no need dapi
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

            if (batch_idx + 1) % args.log_every == 0:
                print(f"[E{epoch} B{batch_idx+1}/{len(train_loader)}] "
                      f"G={loss_g.item():.3f} (gan={loss_g_gan.item():.3f} content={loss_g_content.item():.3f}) "
                      f"D={loss_d.item():.3f} t={time.time()-t0:.0f}s")

        if nan_detected:
            continue

        avg_g = epoch_loss_g / max(n_batches, 1)
        avg_d = epoch_loss_d / max(n_batches, 1)
        elapsed = time.time() - t0

        # Validation
        val_ssim = -1.0
        val_psnr = -1.0
        if val_loader is not None:
            model.eval()
            ssim_sum = 0.0
            psnr_sum = 0.0
            n_val = 0
            with torch.no_grad():
                for vb in val_loader:
                    dapi = vb["dapi"].to(device)
                    real_img = vb["ihc"].to(device)
                    fake_img = model.generator(dapi, dapi)
                    # 计算 SSIM/PSNR（用 [-1,1] 范围，需 clamp + 转到 [0,1]）
                    from .metrics.ssim_psnr import ssim, psnr
                    fake_01 = (fake_img.clamp(-1, 1) + 1) / 2.0
                    real_01 = (real_img.clamp(-1, 1) + 1) / 2.0
                    for f, r in zip(fake_01, real_01):
                        ssim_sum += ssim(f, r)
                        psnr_sum += psnr(f, r)
                    n_val += dapi.size(0)
            val_ssim = ssim_sum / max(n_val, 1)
            val_psnr = psnr_sum / max(n_val, 1)
            print(f"[Epoch {epoch}] avg_G={avg_g:.3f} avg_D={avg_d:.3f} val_ssim={val_ssim:.4f} val_psnr={val_psnr:.2f} t={elapsed:.0f}s")
        else:
            print(f"[Epoch {epoch}] avg_G={avg_g:.3f} avg_D={avg_d:.3f} t={elapsed:.0f}s")

        log_file.write(f'{{"epoch":{epoch},"loss_g":{avg_g:.4f},"loss_d":{avg_d:.4f},"val_ssim":{val_ssim:.4f},"val_psnr":{val_psnr:.2f},"time":{elapsed:.0f}}}\n')
        log_file.flush()

        # Save checkpoints
        ckpt_path = out_dir / f"epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "val_ssim": val_ssim,
            "val_psnr": val_psnr,
        }, ckpt_path)

        if val_ssim > best_ssim:
            best_ssim = val_ssim
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
    print(f"[Done] Best epoch={best_epoch} val_ssim={best_ssim:.4f}")


if __name__ == "__main__":
    main()
