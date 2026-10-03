"""Training script for Marigold-style IHC Latent Diffusion model.

Two-stage training:
1. Full diffusion training with multi-step DDIM
2. One-step Diffusion-FT fine-tuning

Usage:
    # Stage 1: Full diffusion training
    python train_marigold_ihc.py --stage 1 --epochs 50 --batch-size 8
    
    # Stage 2: One-step fine-tuning (Diffusion-FT)
    python train_marigold_ihc.py --stage 2 --checkpoint path/to/stage1.pt --epochs 10
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data.roi_manifest import MARKERS, SEMIFINAL_SEED, digest, selected_names, validate_manifest
from .data.paired_clean import CleanPairedMarkers
from .models.marigold_ihc import MarigoldIHC, MarigoldIHCConfig, build_marigold_ihc


OFFICIAL_JPEG_QUALITY = 100


def seed_all(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def amp_context(device):
    """Get AMP context for mixed precision training."""
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda')


def local_ssim(pred, target):
    """Compute local SSIM for evaluation."""
    if pred.shape != target.shape:
        raise ValueError('Shape mismatch')
    
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, t = pred.float(), target.float()
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        
        # 7x7 average pooling
        kernel_size = 7
        pad = kernel_size // 2
        p_pad = F.pad(p, (pad, pad, pad, pad), mode='reflect')
        t_pad = F.pad(t, (pad, pad, pad, pad), mode='reflect')
        
        # Local statistics
        ones = torch.ones_like(p[:, :1])
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        kernel = kernel.to(p.device)
        
        mp = F.conv2d(p_pad, kernel, groups=p.shape[1]) / 1
        mt = F.conv2d(t_pad, kernel, groups=t.shape[1]) / 1
        
        # Variances and covariance
        pp = F.conv2d(p_pad ** 2, kernel, groups=p.shape[1]) / 1
        tt = F.conv2d(t_pad ** 2, kernel, groups=t.shape[1]) / 1
        pt = F.conv2d(p_pad * t_pad, kernel, groups=t.shape[1]) / 1
        
        vp = pp - mp ** 2
        vt = tt - mt ** 2
        cov = pt - mp * mt
        
        # SSIM
        numerator = (2 * mp * mt + C1) * (2 * cov + C2)
        denominator = (mp ** 2 + mt ** 2 + C1) * (vp + vt + C2)
        ssim_map = numerator / (denominator + 1e-8)
        
        return ssim_map.mean(dim=(-2, -1))


from torch.nn import functional as F


def load_manifest(path, root):
    """Load and validate ROI manifest."""
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_manifest(manifest)
    names = sorted(p.name for p in (Path(root) / 'train' / 'DAPI').glob('*.jpg'))
    if digest(names) != manifest['inventory_names_sha256']:
        raise ValueError('Dataset inventory changed since split generation')
    return manifest


def make_loader(root, names, batch_size, augment=False, cache=True):
    """Create DataLoader for training."""
    reader = CleanPairedMarkers if augment else CleanPairedMarkers
    ds = reader(root, names, augment=augment, cache=cache)
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=augment, 
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(SEMIFINAL_SEED)
    )


def evaluate(model, loader, device, marker_names=MARKERS, num_markers=4):
    """Evaluate model on validation set."""
    model.eval()
    rows = []
    start = time.perf_counter()
    
    marker_names = tuple(marker_names)
    
    with torch.no_grad():
        for x, y, names_batch in loader:
            x = x.to(device)  # DAPI: (B, 1, H, W)
            y = y.to(device)  # IHC: (B, M, H, W)
            
            # Expand DAPI to 3 channels for compatibility
            if x.shape[1] == 1:
                x = x.expand(-1, 3, -1, -1)
            
            # Normalize to [-1, 1]
            x = x * 2 - 1
            y = y * 2 - 1
            
            # Sample for each marker
            B = x.shape[0]
            preds = torch.zeros_like(y)
            
            for marker_idx in range(min(num_markers, y.shape[1])):
                pred = model.sample(x, marker_idx=marker_idx)
                preds[:, marker_idx] = pred[:, 0] if pred.shape[1] > 1 else pred.squeeze(1)
            
            # Convert back to [0, 1]
            preds = (preds + 1) / 2
            targets = (y + 1) / 2
            
            # Compute metrics
            ssim_scores = local_ssim(preds.clamp(0, 1), targets.clamp(0, 1)).cpu().numpy()
            mse = ((preds - targets) ** 2).mean(dim=(-2, -1)).cpu().numpy()
            psnrs = -10 * np.log10(np.maximum(mse, 1e-12))
            
            for name, ss, ps in zip(names_batch, ssim_scores, psnrs):
                rows.append({
                    'name': name,
                    'ssim': ss.tolist() if hasattr(ss, 'tolist') else float(ss),
                    'psnr': ps.tolist() if hasattr(ps, 'tolist') else float(ps)
                })
    
    if not rows:
        raise ValueError('Empty evaluation set')
    
    # Aggregate by marker
    marker_metrics = {}
    for i, m in enumerate(marker_names):
        ssims = [r['ssim'][i] if isinstance(r['ssim'], (list, np.ndarray)) else r['ssim'] 
                 for r in rows]
        psnrs = [r['psnr'][i] if isinstance(r['psnr'], (list, np.ndarray)) else r['psnr'] 
                 for r in rows]
        marker_metrics[m] = {
            'ssim': float(np.mean(ssims)),
            'psnr': float(np.mean(psnrs))
        }
    
    return {
        'count': len(rows),
        'ssim': float(np.mean([marker_metrics[m]['ssim'] for m in marker_names])),
        'psnr': float(np.mean([marker_metrics[m]['psnr'] for m in marker_names])),
        'markers': marker_metrics,
        'seconds': time.perf_counter() - start,
        'rows': rows
    }


def train_stage1(args):
    """Stage 1: Full diffusion training."""
    print("=" * 60)
    print("Stage 1: Full Diffusion Training")
    print("=" * 60)
    
    seed_all(args.seed)
    manifest = load_manifest(args.manifest, args.data_root)
    device = torch.device(args.device)
    
    # Get training/validation names
    train_names = selected_names(manifest['splits']['train'], args.train_limit, args.seed)
    val_names = selected_names(manifest['splits']['val'], args.val_limit, args.seed)
    
    print(f"Train: {len(train_names)}, Val: {len(val_names)}")
    
    train_loader = make_loader(args.data_root, train_names, args.batch_size, augment=True)
    val_loader = make_loader(args.data_root, val_names, args.batch_size, augment=False) if val_names else None
    
    # Build model
    config = MarigoldIHCConfig(
        base_channels=args.base_channels,
        num_markers=len(MARKERS),
        marker_token_dim=args.marker_dim,
        lambda_css=args.css_weight,
        lambda_perceptual=args.perceptual_weight,
    )
    model = MarigoldIHC(config).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # EMA
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    ema_decay = 0.999
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    best_ssim = -math.inf
    steps = 0
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'stage': 1,
            'config': {
                'base_channels': args.base_channels,
                'marker_dim': args.marker_dim,
                'css_weight': args.css_weight,
                'perceptual_weight': args.perceptual_weight,
                'lr': args.lr,
                'batch_size': args.batch_size,
            },
            'markers': list(MARKERS),
            'manifest_sha256': manifest['sha256'],
            'train_names': train_names,
            'val_names': val_names,
        }, f, indent=2)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        t0 = time.perf_counter()
        
        for batch_idx, (dapi, ihc, _) in enumerate(train_loader):
            # Move to device
            dapi = dapi.to(device)  # (B, 1, H, W) -> expand to 3 channels
            ihc = ihc.to(device)    # (B, 4, H, W)
            
            # Expand DAPI to 3 channels and normalize to [-1, 1]
            dapi = dapi.expand(-1, 3, -1, -1) * 2 - 1
            ihc = ihc * 2 - 1
            
            optimizer.zero_grad(set_to_none=True)
            
            # Training: sample random marker
            marker_idx = random.randint(0, len(MARKERS) - 1)
            
            with amp_context(device):
                loss, loss_dict = model.forward_train(ihc, dapi, marker_idx=marker_idx)
            
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at epoch={epoch+1}, batch={batch_idx+1}")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            with torch.no_grad():
                for ema_p, p in zip(ema.parameters(), model.parameters()):
                    ema_p.lerp_(p, 1 - ema_decay)
            
            steps += 1
            epoch_loss += loss.item()
            epoch_steps += 1
            
            if (batch_idx + 1) % 50 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {epoch_loss/epoch_steps:.4f} "
                      f"LR: {lr:.2e} "
                      f"Grad: {grad_norm:.4f}")
        
        # Evaluate
        if val_loader and (epoch + 1) % args.eval_every == 0:
            metrics = evaluate(ema, val_loader, device)
            print(f"Validation @ Epoch {epoch+1}: SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.2f}")
            
            if metrics['ssim'] > best_ssim:
                best_ssim = metrics['ssim']
                torch.save({
                    'epoch': epoch,
                    'steps': steps,
                    'model': model.state_dict(),
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_ssim': best_ssim,
                    'metrics': metrics,
                }, output_dir / 'best.pt')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'steps': steps,
            'model': model.state_dict(),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
        }, output_dir / 'last.pt')
        
        elapsed = time.perf_counter() - t0
        print(f"Epoch {epoch+1} completed in {elapsed:.1f}s, Loss: {epoch_loss/max(epoch_steps,1):.4f}")
    
    print(f"\nStage 1 completed. Best SSIM: {best_ssim:.4f}")
    return best_ssim


def train_stage2(args):
    """Stage 2: One-step Diffusion-FT fine-tuning."""
    print("=" * 60)
    print("Stage 2: One-Step Diffusion-FT Fine-tuning")
    print("=" * 60)
    
    seed_all(args.seed)
    device = torch.device(args.device)
    
    # Load checkpoint from stage 1
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Build model
    config = MarigoldIHCConfig(
        base_channels=args.base_channels,
        num_markers=len(MARKERS),
        marker_token_dim=args.marker_dim,
    )
    model = MarigoldIHC(config).to(device)
    model.load_state_dict(ckpt['ema'], strict=True)
    
    # Freeze most layers, only fine-tune last layers
    # Unfreeze decoder and output
    for name, param in model.named_parameters():
        if 'decoders' in name or 'out_' in name or 'marker_tokens' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Fine-tuning {trainable:,} parameters (out of {sum(p.numel() for p in model.parameters()):,})")
    
    # Optimizer for fine-tuning
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    # Simple training loop for Diffusion-FT
    # Use L1 loss between predicted and target
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (dapi, ihc, _) in enumerate(train_loader):
            dapi = dapi.expand(-1, 3, -1, -1).to(device) * 2 - 1
            ihc = ihc.to(device) * 2 - 1
            
            optimizer.zero_grad(set_to_none=True)
            
            # Sample one marker per batch
            marker_idx = random.randint(0, len(MARKERS) - 1)
            
            with amp_context(device):
                # Use one-step diffusion FT
                pred = model.sample_diffusion_ft(dapi, marker_idx=marker_idx)
                target = ihc[:, marker_idx] if ihc.shape[1] > 1 else ihc.squeeze(1)
                
                # L1 loss for fast convergence
                loss = F.l1_loss(pred, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}")
    
    # Save fine-tuned model
    torch.save({
        'epoch': args.epochs,
        'model': model.state_dict(),
    }, args.output)
    
    print("Stage 2 completed.")


def main():
    parser = argparse.ArgumentParser(description='Train Marigold-style IHC model')
    
    subparsers = parser.add_subparsers(dest='stage', help='Training stage')
    
    # Stage 1 parser
    p1 = subparsers.add_parser('1', help='Stage 1: Full diffusion training')
    p1.add_argument('--data-root', required=True)
    p1.add_argument('--manifest', required=True)
    p1.add_argument('--output', required=True)
    p1.add_argument('--device', default='cuda')
    p1.add_argument('--seed', type=int, default=SEMIFINAL_SEED)
    p1.add_argument('--epochs', type=int, default=50)
    p1.add_argument('--batch-size', type=int, default=8)
    p1.add_argument('--lr', type=float, default=1e-4)
    p1.add_argument('--base-channels', type=int, default=128)
    p1.add_argument('--marker-dim', type=int, default=64)
    p1.add_argument('--css-weight', type=float, default=0.5)
    p1.add_argument('--perceptual-weight', type=float, default=0.1)
    p1.add_argument('--eval-every', type=int, default=5)
    p1.add_argument('--train-limit', type=int, default=0)
    p1.add_argument('--val-limit', type=int, default=0)
    
    # Stage 2 parser
    p2 = subparsers.add_parser('2', help='Stage 2: One-step Diffusion-FT')
    p2.add_argument('--checkpoint', required=True)
    p2.add_argument('--data-root', required=True)
    p2.add_argument('--manifest', required=True)
    p2.add_argument('--output', required=True)
    p2.add_argument('--device', default='cuda')
    p2.add_argument('--seed', type=int, default=SEMIFINAL_SEED)
    p2.add_argument('--epochs', type=int, default=10)
    p2.add_argument('--batch-size', type=int, default=16)
    p2.add_argument('--lr', type=float, default=1e-5)
    p2.add_argument('--base-channels', type=int, default=128)
    p2.add_argument('--marker-dim', type=int, default=64)
    p2.add_argument('--train-limit', type=int, default=0)
    p2.add_argument('--val-limit', type=int, default=0)
    
    args = parser.parse_args()
    
    if args.stage == '1':
        train_stage1(args)
    elif args.stage == '2':
        train_stage2(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
