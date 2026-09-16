"""VSMT (CVPR 2026) 继续训练脚本（可作为脚本直接运行）

用法：
    python resume_train_vsmt.py --marker HLA-DR --epochs 12 --lr 1e-5
   或：
    python -m src.resume_train_vsmt --marker HLA-DR --epochs 12 --lr 1e-5

⚠️ 直接 python 运行时需要 sys.path 注入（脚本里已处理）。
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.data.dataset import DAPItoIHCDataset
from src.models.pix2pix_gan import build_pix2pix_model
from src.models.losses import CombinedLoss, GANLoss
from src.models.vsmt import (
    VSMTLosses,
    MultiTaskFeatureExtractor,
    TaskGapAlignment,
)


def find_latest_ckpt(marker: str, prefer_best: bool = True):
    ck_root = ROOT / "checkpoints"
    dirs = sorted(
        (d for d in ck_root.glob(f"pix2pix_v2_{marker}_*") if d.is_dir()),
        key=lambda d: int(d.name.rsplit("_", 1)[1]),
        reverse=True,
    )
    for d in dirs:
        if prefer_best and (d / "best.pt").exists():
            return d / "best.pt"
        epoch_files = sorted(d.glob("epoch*.pt"), key=lambda p: int(p.stem[5:]), reverse=True)
        if epoch_files:
            return epoch_files[0]
        if (d / "final.pt").exists():
            return d / "final.pt"
    return None


class UNetWithHook(nn.Module):
    """UNetGenerator 包装：暴露中间层特征供 Msf 使用"""

    def __init__(self, base_unet: nn.Module):
        super().__init__()
        self.unet = base_unet
        self._feat_2 = None  # encoder 第 2 层 (256 ch, 32x32)
        self._feat_3 = None  # encoder 第 3 层 (512 ch, 16x16)
        # 注册 hook
        self.unet.encoder[1].register_forward_hook(self._hook_layer2)
        self.unet.encoder[2].register_forward_hook(self._hook_layer3)

    def _hook_layer2(self, module, input, output):
        self._feat_2 = output  # (B, 128, 64, 64)

    def _hook_layer3(self, module, input, output):
        self._feat_3 = output  # (B, 256, 32, 32)

    def forward(self, x, dapi=None):
        return self.unet(x, dapi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--marker", type=str, default="HLA-DR")
    p.add_argument("--epochs", type=int, default=12, help="每个 marker 续训 epoch 数")
    p.add_argument("--lr", type=float, default=1e-5, help="极小学习率避免破坏已有知识")
    p.add_argument("--batch_size", type=int, default=12, help="VSMT 需要更多显存（多了 ResNet + Msf）")
    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--lambda_ssim", type=float, default=50.0)
    p.add_argument("--lambda_sel", type=float, default=1.0, help="SEL contrastive 损失权重")
    p.add_argument("--lambda_apm", type=float, default=1.0, help="APM rearrangement 损失权重")
    p.add_argument("--lambda_mta", type=float, default=10.0, help="Task-gap alignment 特征注入强度")
    p.add_argument("--sel_warmup_epochs", type=int, default=2, help="前 N 个 epoch 只训练 Msf")
    p.add_argument("--data_root", type=str,
                   default=r"E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--save_every", type=int, default=2)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--prefer_best", action="store_true", default=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VSMT] marker={args.marker} device={device} epochs={args.epochs} bs={args.batch_size}")
    print(f"[VSMT] lambdas: sel={args.lambda_sel} apm={args.lambda_apm} mta={args.lambda_mta}")

    # ============ 数据 ============
    train_ds_full = DAPItoIHCDataset(
        root=args.data_root,
        marker=args.marker,
        split="train",
        patch_size=256,
        augment=True,
    )
    val_size = max(1, int(len(train_ds_full) * args.val_split))
    val_size = min(val_size, len(train_ds_full) - 1)
    import random
    rng = random.Random(42)
    val_indices = sorted(rng.sample(range(len(train_ds_full)), val_size))
    train_indices = [i for i in range(len(train_ds_full)) if i not in set(val_indices)]
    val_ds = Subset(DAPItoIHCDataset(
        root=args.data_root, marker=args.marker, split="train",
        patch_size=256, augment=False,
    ), val_indices)
    train_ds = Subset(train_ds_full, train_indices)
    print(f"[VSMT] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ============ 模型 ============
    model = build_pix2pix_model(input_channels=3, cond_channels=3, output_channels=3, base_filters=64).to(device)
    # 用 hook 包装生成器，提取中间层特征
    gen_wrapped = UNetWithHook(model.generator).to(device)
    model.generator = gen_wrapped.unet  # 保持外部接口不变

    # VSMT 模块
    # Msf 输入维度：encoder 第 2 层输出 128 通道（ConvBlock(64,128)）
    vsmt_losses = VSMTLosses(msf_in_channels=128, msf_hidden=128, num_clusters=3,
                              lambda_sel=args.lambda_sel, lambda_apm=args.lambda_apm).to(device)
    # Mta 输入维度：多任务特征拼接 (cls 128 + recon 256 = 384)
    mta = TaskGapAlignment(in_dim=384, out_dim=512, hidden=256).to(device)
    # 多任务特征提取器
    mtf = MultiTaskFeatureExtractor().to(device)

    # 加载 best.pt（兼容旧 checkpoint，只载入 G/D 权重）
    ckpt_path = find_latest_ckpt(args.marker, prefer_best=args.prefer_best)
    if ckpt_path is not None:
        print(f"[VSMT] resume from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    else:
        print(f"[VSMT] no checkpoint found for {args.marker}, training from scratch")

    # 冻结 Gvis 主体（VSMT 论文策略：只微调辅助模块）
    # 但 discriminator 不冻结——需要继续学对抗
    if ckpt_path is not None:
        for p in model.generator.parameters():
            p.requires_grad = False
        print("[VSMT] Gvis frozen (have ckpt)")
    else:
        print("[VSMT] Gvis trainable (no ckpt, train from scratch)")
    for p in model.discriminator.parameters():
        p.requires_grad = True

    # 优化器：只优化 Msf + Mta + Discriminator
    optimizer_g = optim.Adam(
        list(vsmt_losses.msf.parameters()) + list(mta.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_d = optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # 损失
    image_criterion = CombinedLoss(l1_weight=args.lambda_l1, ssim_weight=args.lambda_ssim,
                                    edge_weight=5.0, css_weight=0.0).to(device)  # 关掉 CSS，让 VSMT 自己管
    gan_criterion = GANLoss(loss_type='lsgan').to(device)

    # ============ 训练 ============
    timestamp = int(time.time())
    out_dir = Path(f"./checkpoints/pix2pix_vsmt_{args.marker}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")
    best_ssim = -1.0
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        vsmt_losses.train()
        mta.train()
        t0 = time.time()
        ep_g = ep_d = ep_sel = ep_apm = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            dapi = batch["dapi"].to(device, non_blocking=True)
            real_img = batch["ihc"].to(device, non_blocking=True)

            # ====== 1. 训练 Msf (SEL 对比损失) ======
            optimizer_g.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_img, fake_feat = model.generator(dapi, dapi, return_feats=True)
            sel_loss = vsmt_losses.forward_sel(fake_feat.detach())
            sel_loss.backward()
            optimizer_g.step()
            ep_sel += sel_loss.item()

            # ====== 2. 训练 Discriminator ======
            optimizer_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_img = model.generator(dapi, dapi)
            real_pred = model.discriminator(real_img, dapi)
            loss_d_real = gan_criterion(real_pred, True)
            fake_pred = model.discriminator(fake_img.detach(), dapi)
            loss_d_fake = gan_criterion(fake_pred, False)
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()
            ep_d += loss_d.item()

            # ====== 3. 训练 Mta (Task-gap alignment) ======
            # 跳过 warmup（让 Msf 先稳定）
            if epoch >= args.sel_warmup_epochs:
                optimizer_g.zero_grad(set_to_none=True)
                # 1) 生成虚拟 IHC（不计算 Gvis 梯度）
                with torch.no_grad():
                    fake_img = model.generator(dapi, dapi)
                    feat_virt_2 = gen_wrapped._feat_2  # encoder 第 2 层特征
                # 2) 把 IHC 也喂给 Gvis（用真实 IHC 重建它自己）→ 提取 real 特征
                #    论文用 Gvis(y) 重建 real IHC，简化：直接用 fake_img 的特征近似
                #    （论文注释：rebuilt real 用于稳定聚类；这里为了节省算力，跳过）
                # 3) 计算 IHC 多任务特征
                cls_feat_real, recon_feat_real = mtf(real_img)  # (B,128,H/8,W/8), (B,256,H/16,W/16)
                # 下采样到与 Msf 输出同尺寸
                cls_down = F.adaptive_avg_pool2d(cls_feat_real, output_size=feat_virt_2.shape[-2:])
                recon_down = F.adaptive_avg_pool2d(recon_feat_real, output_size=feat_virt_2.shape[-2:])
                multitask = torch.cat([cls_down, recon_down], dim=1)  # (B, 384, H', W')

                # 4) APM 重排 + 一致性损失
                # 简化：用 fake_img 自身当作 rebuilt real（论文里 yr = Gvis(y)）
                # 这里直接用 real_img 通过 Gvis 的 encoder 提取 feat_real_2
                with torch.no_grad():
                    _ = model.generator(real_img, dapi)
                    feat_real_2 = gen_wrapped._feat_2

                apm_loss, aligned_feat = vsmt_losses.forward_apm(
                    feat_real_in=feat_real_2,
                    feat_virt_in=feat_virt_2,
                    feat_real_multitask=multitask,
                )
                ep_apm += apm_loss.item()

                # 5) Mta loss：用 aligned 特征逼近 virtual 特征（语义保留）
                B, C, H, W = feat_virt_2.shape
                mt_flat = aligned_feat  # (B, N, 384)
                mta_feat = mta(mt_flat)  # (B, N, 512)
                target_flat = feat_virt_2.permute(0, 2, 3, 1).reshape(B, H * W, -1)
                # 投影到 512 维（如果不一致）
                if mta_feat.shape[-1] != target_flat.shape[-1]:
                    target_proj = target_flat.mean(dim=-1, keepdim=True).expand_as(mta_feat)
                else:
                    target_proj = target_flat
                mta_loss = F.mse_loss(mta_feat, target_proj.detach())
                ep_apm += mta_loss.item()

                # 6) 整体 G 损失：内容损失 + mta 注入
                #    简化版：Mta 特征作为额外监督信号（不直接注入 Gvis 中间层，避免架构改动）
                #    论文式注入需要改 forward，这里采用 proxy loss 形式
                mta_total = apm_loss + args.lambda_mta * mta_loss

                # 内容损失（仍用 Gvis 主体做监督）
                fake_img2 = model.generator(dapi, dapi)
                content_loss = image_criterion(fake_img2, real_img, dapi=None)  # CSS 关掉了
                loss_g_total = content_loss + mta_total
                if torch.isfinite(loss_g_total):
                    loss_g_total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(vsmt_losses.msf.parameters()) + list(mta.parameters()),
                        max_norm=1.0
                    )
                    optimizer_g.step()
                ep_g += content_loss.item()

            n_batches += 1
            if batch_idx % args.log_every == 0:
                print(f"[E{epoch} B{batch_idx}/{len(train_loader)}] "
                      f"D={loss_d.item():.3f} SEL={sel_loss.item():.3f} "
                      f"APM={ep_apm/max(n_batches,1):.3f}")

        avg_g = ep_g / max(n_batches, 1)
        avg_d = ep_d / max(n_batches, 1)
        avg_sel = ep_sel / max(n_batches, 1)
        avg_apm = ep_apm / max(n_batches, 1)
        elapsed = time.time() - t0
        log_entry = {
            "epoch": epoch, "loss_g": avg_g, "loss_d": avg_d,
            "sel": avg_sel, "apm": avg_apm, "elapsed": elapsed,
        }
        print(f"[Epoch {epoch}] G={avg_g:.4f} D={avg_d:.4f} SEL={avg_sel:.4f} APM={avg_apm:.4f} time={elapsed:.1f}s")
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        if (epoch + 1) % args.save_every == 0:
            ckpt_path_out = out_dir / f"epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "vsmt_msf": vsmt_losses.msf.state_dict(),
                "mta": mta.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
            }, ckpt_path_out)

        # 验证
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
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "vsmt_msf": vsmt_losses.msf.state_dict(),
                "mta": mta.state_dict(),
                "val_ssim": vm["ssim"],
            }, out_dir / "best.pt")
            print(f"[Best] new best SSIM={vm['ssim']:.4f} @ epoch {epoch}")

    final_path = out_dir / "final.pt"
    torch.save({
        "epoch": args.epochs - 1,
        "model": model.state_dict(),
        "vsmt_msf": vsmt_losses.msf.state_dict(),
        "mta": mta.state_dict(),
    }, final_path)
    print(f"[VSMT] done -> {final_path}")
    if best_epoch >= 0:
        print(f"[VSMT] best.pt SSIM={best_ssim:.4f} @ epoch {best_epoch}")
    log_file.close()


if __name__ == "__main__":
    main()
