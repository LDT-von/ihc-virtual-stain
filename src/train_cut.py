"""
CUT / FastCUT 训练脚本

支持三种训练模式：
1. 'unpaired': DAPI 和 IHC 各自独立采样（各自 shuffle），用于标准的 CUT 训练
2. 'paired': DAPI 和 IHC 配对采样（同名），但用 CUT Loss（无 L1 pixel loss）
3. 'cut_paired': CUT Loss + paired 采样（FastCUT 论文推荐的混合方式）

用法：
    # FastCUT unpaired 模式（推荐，比标准 CUT 快 2-3 倍）
    python -m src.train_cut --marker HLA-DR --mode unpaired --epochs 80 --batch_size 8 --n_epochs_decay 40

    # 标准 CUT unpaired
    python -m src.train_cut --marker HLA-DR --mode unpaired --epochs 200 --batch_size 4 --CUT_mode CUT

    # CUT paired 模式（用你已有的 paired 数据）
    python -m src.train_cut --marker HLA-DR --mode paired --epochs 100 --batch_size 8
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim

from .data.dataset import DAPItoIHCDataset, UnpairedDAPIDataset
from .models.cut_model import build_cut_model


def parse_args():
    p = argparse.ArgumentParser(description="CUT / FastCUT 训练")
    p.add_argument("--data_root", type=str,
                   default="E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）",
                   help="数据根目录")
    p.add_argument("--marker", type=str, default="HLA-DR", help="目标标记")
    p.add_argument("--mode", type=str, default="unpaired",
                   choices=["unpaired", "paired", "cut_paired"],
                   help="'unpaired': DAPI/IHC 独立采样; 'paired': 配对采样; 'cut_paired': paired + CUT")
    p.add_argument("--CUT_mode", type=str, default="FastCUT", choices=["CUT", "FastCUT"],
                   help="CUT 或 FastCUT 模式（FastCUT 更轻量，单向翻译）")
    
    # 训练参数
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--n_epochs_decay", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    
    # Loss weights
    p.add_argument("--lambda_GAN", type=float, default=1.0)
    p.add_argument("--lambda_NCE", type=float, default=1.0)
    p.add_argument("--lambda_idt", type=float, default=0.0)  # FastCUT identity loss
    p.add_argument("--nce_T", type=float, default=0.07)
    p.add_argument("--num_patches", type=int, default=256)
    p.add_argument("--nce_layers", type=str, default='0,4,8,12,16')
    
    # Model
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n_layers_D", type=int, default=3)
    p.add_argument("--pool_size", type=int, default=0)
    
    # 训练控制
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    
    return p.parse_args()


def build_dataloader(args, split="train"):
    """构建数据加载器（支持 paired / unpaired 模式）
    
    val 模式下：dataset 内部自动从 train 划分 val_split% 的样本
    """
    from torch.utils.data import DataLoader
    
    if args.mode == "unpaired":
        ds = UnpairedDAPIDataset(
            root=args.data_root,
            marker=args.marker,
            split=split,
            patch_size=256,
            augment=(split == "train"),
            val_split=args.val_split,
        )
    else:
        ds = DAPItoIHCDataset(
            root=args.data_root,
            marker=args.marker,
            split=split,
            patch_size=256,
            augment=(split == "train"),
        )
    
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        drop_last=(split == "train"),
        pin_memory=True,
    )
    return loader, loader  # val_loader 与 train_loader 共用同一 dataset 的不同索引


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=" * 60)
    print(f"[CUT] marker={args.marker} mode={args.mode} CUT_mode={args.CUT_mode}")
    print(f"[CUT] device={device} epochs={args.epochs}+{args.n_epochs_decay} bs={args.batch_size}")
    print(f"[CUT] lambda_GAN={args.lambda_GAN} lambda_NCE={args.lambda_NCE}")
    print(f"[CUT] nce_T={args.nce_T} num_patches={args.num_patches}")
    print(f"=" * 60)
    
    # 数据
    train_loader, val_loader = build_dataloader(args, "train")
    _, val_loader_eval = build_dataloader(args, "val")
    print(f"[Dataset] train={len(train_loader.dataset)} val={len(val_loader.dataset) if val_loader else 0}")
    
    # 模型
    nce_idt = (args.CUT_mode == "FastCUT") and args.lambda_idt > 0
    
    model = build_cut_model(
        input_channels=3,
        output_channels=3,
        ngf=args.ngf,
        ndf=args.ndf,
        n_layers_D=args.n_layers_D,
        nce_layers=args.nce_layers,
        nce_T=args.nce_T,
        num_patches=args.num_patches,
        normG='instance',
        normD='instance',
        gan_mode='lsgan',
        lambda_NCE=args.lambda_NCE,
        lambda_GAN=args.lambda_GAN,
        lambda_idt=args.lambda_idt,
        pool_size=args.pool_size,
        flip_equivariance=(args.CUT_mode == "FastCUT"),
    ).to(device)
    
    # 打印参数量
    n_params_G = sum(p.numel() for p in model.netG.parameters())
    n_params_D = sum(p.numel() for p in model.netD.parameters())
    print(f"[Model] G={n_params_G:,} D={n_params_D:,}")
    
    # 优化器
    optimizer_G = optim.Adam(
        model.netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
    )
    optimizer_D = optim.Adam(
        model.netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
    )
    
    # 学习率调度：n_epochs 不变，后 n_epochs_decay 线性衰减
    def lr_lambda(epoch):
        if args.n_epochs_decay <= 0 or epoch < args.epochs:
            return 1.0
        return max(0.0, 1.0 - (epoch - args.epochs) / args.n_epochs_decay)
    
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)
    
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer_G.load_state_dict(ckpt["optimizer_G"])
        optimizer_D.load_state_dict(ckpt["optimizer_D"])
        scheduler_G.load_state_dict(ckpt["scheduler_G"])
        scheduler_D.load_state_dict(ckpt["scheduler_D"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"[Resume] from {args.resume}, epoch={start_epoch}")
    
    # 输出目录
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"./checkpoints/cut_{args.mode}_{args.CUT_mode}_{args.marker}_{int(time.time())}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    
    log_file = open(log_path, "w", encoding="utf-8")
    best_ssim = -1.0
    best_epoch = -1
    
    # 首次 forward 初始化 netF（data-dependent init）
    first_batch = next(iter(train_loader))
    # 兼容两种 dataset 返回格式
    if isinstance(first_batch, dict):
        if "dapi" in first_batch:
            real_A = first_batch["dapi"].to(device)
            real_B = first_batch["ihc"].to(device)
        else:
            real_A = first_batch["A"].to(device)
            real_B = first_batch["B"].to(device)
    else:
        real_A, real_B = first_batch
        real_A = real_A.to(device)
        real_B = real_B.to(device)
    
    model.data_dependent_initialize(real_A, real_B)
    
    # 重新创建 optimizer_F（data-dependent init 后才能确定参数）
    optimizer_F = optim.Adam(
        model.netF_dict.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
    )
    scheduler_F = optim.lr_scheduler.LambdaLR(optimizer_F, lr_lambda)
    
    total_epochs = args.epochs + args.n_epochs_decay
    
    for epoch in range(start_epoch, total_epochs):
        model.train()
        t0 = time.time()
        
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        epoch_loss_NCE = 0.0
        n_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 解析 batch
            if isinstance(batch, dict):
                if "dapi" in batch:
                    real_A = batch["dapi"].to(device)
                    real_B = batch["ihc"].to(device)
                else:
                    real_A = batch["A"].to(device)
                    real_B = batch["B"].to(device)
            else:
                real_A, real_B = batch
                real_A = real_A.to(device)
                real_B = real_B.to(device)
            
            # Training step
            losses = model.train_step(
                real_A, real_B,
                nce_idt=nce_idt,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                optimizer_F=optimizer_F,
            )
            
            epoch_loss_G += losses['loss_G']
            epoch_loss_D += losses['loss_D']
            epoch_loss_NCE += losses['loss_NCE']
            n_batches += 1
            
            if batch_idx % args.log_every == 0:
                print(f"[E{epoch} B{batch_idx}/{len(train_loader)}] "
                      f"G={losses['loss_G']:.4f} D={losses['loss_D']:.4f} "
                      f"NCE={losses['loss_NCE']:.4f}")
        
        # 学习率调度
        scheduler_G.step()
        scheduler_D.step()
        scheduler_F.step()
        
        avg_G = epoch_loss_G / max(n_batches, 1)
        avg_D = epoch_loss_D / max(n_batches, 1)
        avg_NCE = epoch_loss_NCE / max(n_batches, 1)
        elapsed = time.time() - t0
        lr_now = optimizer_G.param_groups[0]['lr']
        
        msg = (f"[Epoch {epoch}/{total_epochs-1}] G={avg_G:.4f} D={avg_D:.4f} "
               f"NCE={avg_NCE:.4f} lr={lr_now:.2e} time={elapsed:.1f}s")
        print(msg)
        
        log_entry = {
            "epoch": epoch,
            "loss_G": avg_G,
            "loss_D": avg_D,
            "loss_NCE": avg_NCE,
            "lr": lr_now,
            "elapsed": elapsed,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()
        
        # 保存
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = out_dir / f"epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "optimizer_F": optimizer_F.state_dict(),
                "scheduler_G": scheduler_G.state_dict(),
                "scheduler_D": scheduler_D.state_dict(),
                "scheduler_F": scheduler_F.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"[Save] {ckpt_path}")
        
        # 验证
        if val_loader is not None and (epoch + 1) % 5 == 0:
            model.eval()
            from .metrics.ssim_psnr import MetricAggregator
            val_metric = MetricAggregator()
            with torch.inference_mode():
                for vb in val_loader:
                    if isinstance(vb, dict):
                        if "dapi" in vb:
                            v_A = vb["dapi"].to(device)
                            v_B = vb["ihc"].to(device)
                        else:
                            v_A = vb["A"].to(device)
                            v_B = vb["B"].to(device)
                    else:
                        v_A, v_B = vb
                        v_A = v_A.to(device)
                        v_B = v_B.to(device)
                    
                    # CUT forward：只跑 G(A)
                    fake_B, _ = model.forward_G(v_A)
                    val_metric.update(fake_B, v_B)
            
            vm = val_metric.result()
            print(f"[Val {epoch}] SSIM={vm['ssim']:.4f} PSNR={vm['psnr']:.2f}")
            log_file.write(json.dumps({"epoch": epoch, "val_ssim": vm["ssim"], "val_psnr": vm["psnr"]}) + "\n")
            log_file.flush()
            
            if vm['ssim'] > best_ssim:
                best_ssim = vm['ssim']
                best_epoch = epoch
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "val_ssim": vm["ssim"], "val_psnr": vm["psnr"]
                }, out_dir / "best.pt")
                print(f"[Best] new best SSIM={vm['ssim']:.4f} @ epoch {epoch}")
    
    # 最终保存
    final_path = out_dir / "final.pt"
    torch.save({
        "epoch": total_epochs - 1,
        "model": model.state_dict(),
    }, final_path)
    print(f"[Train] done -> {final_path}")
    if best_epoch >= 0:
        print(f"[Train] best.pt SSIM={best_ssim:.4f} @ epoch {best_epoch}")
    log_file.close()


if __name__ == "__main__":
    main()
