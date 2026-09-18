"""
Stable Diffusion 虚拟染色模型训练脚本

支持功能:
- 像素空间扩散 (无需 VAE)
- Classifier-Free Guidance 训练
- EMA 指数移动平均
- 验证集 SSIM/PSNR 评测
- 断点续训

Usage:
    python scripts/train_sd.py --data-root /path/to/data --epochs 60 --batch-size 4
"""

import argparse
import builtins
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# 路径设置
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# 解决 Windows 编码问题
import builtins
_original_print = builtins.print

def _safe_print(*a, **k):
    k.setdefault('flush', True)
    try:
        _original_print(*a, **k)
    except UnicodeEncodeError:
        safe = [str(x).encode('gbk', errors='replace').decode('gbk') for x in a]
        _original_print(*safe, **k)

builtins.print = _safe_print

from src.models.stable_diffusion import SimpleVirtualStainDiffusion, SimpleDiffusionConfig
from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import to_uint8, ssim_gpu_single, psnr_gpu_single


# ============================================================================
# EMA
# ============================================================================

class EMA:
    """
    指数移动平均
    
    Usage:
        ema = EMA(model, decay=0.9999)
        ema.update()
        ema.apply(model)  # 用 EMA 参数覆盖原模型
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# 数据加载
# ============================================================================

def build_dataloader(
    data_root: Path,
    marker: str,
    split: str,
    batch_size: int,
    patch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = True,
):
    """构建数据加载器"""
    dataset = DAPItoIHCDataset(
        root=data_root,
        marker=marker,
        split=split,
        patch_size=patch_size,
        augment=(split == 'train'),
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=(split == 'train'),
        pin_memory=True,
        collate_fn=collate_batch,
    )
    
    return loader, len(dataset)


def collate_batch(batch):
    """批处理整理"""
    return {
        'dapi': torch.stack([b['dapi'] for b in batch]),
        'ihc': torch.stack([b['ihc'] for b in batch]),
        'name': [b['name'] for b in batch],
    }


# ============================================================================
# 评估
# ============================================================================

def evaluate(
    model: SimpleVirtualStainDiffusion,
    val_loader: DataLoader,
    device: torch.device,
    num_eval_samples: int = 100,
) -> dict:
    """
    评估模型
    
    返回 SSIM 和 PSNR
    """
    model.eval()
    
    ssim_sum, psnr_sum, n = 0.0, 0.0, 0
    
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            
            # 采样生成
            fake = model.sample(
                dapi, 
                shape=(dapi.shape[0], 3, dapi.shape[2], dapi.shape[3]),
                num_steps=50,
                method='ddim',
            )
            
            # 计算指标
            for i in range(fake.shape[0]):
                ssim_val = ssim_gpu_single(fake[i], real[i])
                psnr_val = psnr_gpu_single(fake[i], real[i])
                ssim_sum += ssim_val
                psnr_sum += psnr_val
                n += 1
            
            if n >= num_eval_samples:
                break
    
    model.train()
    
    return {
        'ssim': ssim_sum / n if n > 0 else 0.0,
        'psnr': psnr_sum / n if n > 0 else 0.0,
        'n': n,
    }


def evaluate_fast(
    model: SimpleVirtualStainDiffusion,
    val_loader: DataLoader,
    device: torch.device,
    num_eval_samples: int = 50,
) -> dict:
    """
    快速评估 (减少采样步数)
    """
    model.eval()
    
    ssim_sum, psnr_sum, n = 0.0, 0.0, 0
    
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            
            # 采样生成 (减少步数加快评估)
            fake = model.sample(
                dapi,
                shape=(dapi.shape[0], 3, dapi.shape[2], dapi.shape[3]),
                num_steps=20,
                method='ddim',
            )
            
            # 确保 fake 和 real 尺寸一致
            if fake.shape != real.shape:
                fake = F.interpolate(fake, size=real.shape[-2:], mode='bilinear', align_corners=False)
            
            # 计算指标
            for i in range(fake.shape[0]):
                ssim_val = ssim_gpu_single(fake[i], real[i])
                psnr_val = psnr_gpu_single(fake[i], real[i])
                ssim_sum += ssim_val
                psnr_sum += psnr_val
                n += 1
            
            if n >= num_eval_samples:
                break
    
    model.train()
    
    return {
        'ssim': ssim_sum / n if n > 0 else 0.0,
        'psnr': psnr_sum / n if n > 0 else 0.0,
        'n': n,
    }


# ============================================================================
# 训练
# ============================================================================

def train_epoch(
    model: SimpleVirtualStainDiffusion,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    cond_drop_prob: float = 0.1,
    grad_clip: float = 1.0,
) -> dict:
    """
    训练一个 epoch
    """
    model.train()
    
    loss_sum, n_b = 0.0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        dapi = batch['dapi'].to(device)
        real = batch['ihc'].to(device)
        
        # 前向传播
        optimizer.zero_grad(set_to_none=True)
        loss = model.forward_train(real, dapi, cond_drop_prob=cond_drop_prob)
        
        # 检查 loss 有效性
        if not torch.isfinite(loss):
            print(f"[Warning] Non-finite loss: {loss.item()}, skipping...")
            continue
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        loss_sum += loss.item()
        n_b += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {
        'loss': loss_sum / max(n_b, 1),
        'n_batches': n_b,
    }


def main():
    parser = argparse.ArgumentParser(description='Stable Diffusion 虚拟染色训练')
    
    # 数据参数
    parser.add_argument('--data-root', type=str, 
                       default='E:/aic/ihc-data',
                       help='数据根目录')
    parser.add_argument('--marker', type=str, default='HLA-DR', 
                       help='IHC marker 名称')
    parser.add_argument('--patch-size', type=int, default=256,
                       help='Patch 尺寸')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=60,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='梯度裁剪')
    parser.add_argument('--cond-drop-prob', type=float, default=0.1,
                       help='CFG 条件丢弃概率')
    
    # 模型参数
    parser.add_argument('--model-channels', type=int, default=128,
                       help='UNet 基础通道数')
    parser.add_argument('--latent-channels', type=int, default=4,
                       help='VAE latent 通道数')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout')
    
    # 采样参数
    parser.add_argument('--num-sampling-steps', type=int, default=50,
                       help='DDIM 采样步数')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
                       help='Classifier-Free Guidance 强度')
    
    # EMA 参数
    parser.add_argument('--ema-decay', type=float, default=0.9999,
                       help='EMA decay')
    parser.add_argument('--ema-update-every', type=int, default=1,
                       help='EMA 更新频率')
    
    # 其他
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载 worker 数')
    parser.add_argument('--val-every', type=int, default=1,
                       help='每多少个 epoch 验证一次')
    parser.add_argument('--save-every', type=int, default=5,
                       help='每多少个 epoch 保存一次')
    parser.add_argument('--output-dir', type=str, default='outputs/sd_v1',
                       help='输出目录')
    parser.add_argument('--resume', type=str, default='',
                       help='恢复训练的 checkpoint 路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--use-val-split', action='store_true',
                       help='使用 val 目录划分验证集')
    parser.add_argument('--eval-samples', type=int, default=50,
                       help='验证时评估的样本数')
    
    args = parser.parse_args()
    
    # 环境变量支持
    env_root = os.environ.get('IHC_DATA_ROOT')
    if env_root:
        args.data_root = env_root
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(f"[Train SD] Virtual Staining with Stable Diffusion")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Marker: {args.marker}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Model channels: {args.model_channels}")
    print(f"Latent channels: {args.latent_channels}")
    print("=" * 70)
    
    # 数据
    data_root = Path(args.data_root)
    train_loader, n_train = build_dataloader(
        data_root, args.marker, 'train',
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader, n_val = build_dataloader(
        data_root, args.marker, 'val',
        batch_size=2,  # 验证用小 batch
        patch_size=args.patch_size,
        num_workers=0,
        shuffle=False,
    )
    
    print(f"Train samples: {n_train}, Val samples: {n_val}")
    print(f"Train batches: {len(train_loader)}")
    
    # 模型
    config = SimpleDiffusionConfig(
        model_channels=args.model_channels,
        dropout=args.dropout,
        num_sampling_steps=args.num_sampling_steps,
    )
    model = SimpleVirtualStainDiffusion(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} params, {n_trainable:,} trainable")
    
    # EMA
    ema = EMA(model, decay=args.ema_decay)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader) * 5,  # 5 个 epoch 重启一次
        T_mult=2,
    )
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查点
    start_epoch = 0
    best_ssim = 0.0
    best_epoch = -1
    history = []
    
    if args.resume:
        print(f"[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_ssim = ckpt.get('best_ssim', 0.0)
        best_epoch = ckpt.get('best_epoch', -1)
        history = ckpt.get('history', [])
        print(f"[Resume] Start from epoch {start_epoch}, best SSIM: {best_ssim:.4f}")
    
    # 初始评估
    if start_epoch == 0:
        print("[Eval] Initial evaluation...")
        metrics = evaluate_fast(model, val_loader, device, num_eval_samples=args.eval_samples)
        print(f"[Init] SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.2f}")
        best_ssim = metrics['ssim']
    
    # 训练循环
    total_start = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch,
            cond_drop_prob=args.cond_drop_prob,
            grad_clip=args.grad_clip,
        )
        
        # 更新学习率
        scheduler.step()
        
        # EMA 更新
        if (epoch + 1) % args.ema_update_every == 0:
            ema.update()
        
        epoch_time = time.time() - epoch_start
        
        # 验证
        do_val = (epoch + 1) % args.val_every == 0
        val_metrics = {'ssim': 0.0, 'psnr': 0.0}
        
        if do_val:
            # 使用 EMA 模型验证
            ema.apply()
            val_metrics = evaluate_fast(model, val_loader, device, num_eval_samples=args.eval_samples)
            ema.restore()
            
            print(f"[Epoch {epoch}] Loss: {train_metrics['loss']:.4f} | "
                  f"Val SSIM: {val_metrics['ssim']:.4f} | "
                  f"Val PSNR: {val_metrics['psnr']:.2f} | "
                  f"Time: {epoch_time:.0f}s")
        else:
            print(f"[Epoch {epoch}] Loss: {train_metrics['loss']:.4f} | "
                  f"Time: {epoch_time:.0f}s")
        
        # 记录历史
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_ssim': val_metrics['ssim'],
            'val_psnr': val_metrics['psnr'],
            'lr': optimizer.param_groups[0]['lr'],
        })
        
        # 保存检查点
        is_best = val_metrics['ssim'] > best_ssim
        if is_best:
            best_ssim = val_metrics['ssim']
            best_epoch = epoch
        
        if is_best or (epoch + 1) % args.save_every == 0:
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': config,
                'best_ssim': best_ssim,
                'best_epoch': best_epoch,
                'history': history,
            }
            
            if is_best:
                torch.save(ckpt, output_dir / 'best.pt')
                print(f"  [Best] SSIM: {best_ssim:.4f} (epoch {best_epoch})")
            
            if (epoch + 1) % args.save_every == 0:
                torch.save(ckpt, output_dir / f'epoch{epoch}.pt')
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, output_dir / 'latest.pt')
    
    # 训练结束
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("[Training Complete]")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Best SSIM: {best_ssim:.4f} (epoch {best_epoch})")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # 保存训练报告
    report_path = output_dir / 'training_report.txt'
    with open(report_path, 'w') as f:
        f.write("Stable Diffusion Virtual Staining Training Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Marker: {args.marker}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Model channels: {args.model_channels}\n")
        f.write(f"Latent channels: {args.latent_channels}\n")
        f.write(f"Total training time: {total_time / 3600:.2f} hours\n")
        f.write(f"Best validation SSIM: {best_ssim:.4f}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write("\nHistory:\n")
        for h in history:
            f.write(f"  Epoch {h['epoch']}: loss={h['train_loss']:.4f}, "
                   f"val_ssim={h['val_ssim']:.4f}, val_psnr={h['val_psnr']:.2f}\n")
    
    print(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
