"""Training script for Enhanced UNet++ IHC model.

This is a simpler, more efficient training script compared to the diffusion approach.

Usage:
    python train_enhanced_unet.py --data-root DATA_ROOT --manifest MANIFEST --output checkpoints/enhanced_unet
"""
import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data.roi_manifest import MARKERS, SEMIFINAL_SEED, digest, selected_names, validate_manifest
from .data.paired_clean import CleanPairedMarkers
from .models.enhanced_unetpp import (
    EnhancedUNetPlusPlus,
    combined_loss,
    build_enhanced_unet,
)


OFFICIAL_JPEG_QUALITY = 100


def seed_all(seed: int):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_context(device):
    """Get AMP context."""
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda')


def local_ssim(pred, target, window_size=7):
    """Compute local SSIM for evaluation."""
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, t = pred.float(), target.float()
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        
        pad = window_size // 2
        p_pad = torch.nn.functional.pad(p, (pad, pad, pad, pad), mode='reflect')
        t_pad = torch.nn.functional.pad(t, (pad, pad, pad, pad), mode='reflect')
        
        kernel = torch.ones(window_size, window_size, device=pred.device) / (window_size ** 2)
        
        # Local means
        mu_p = torch.nn.functional.conv2d(p_pad, kernel.expand(p.shape[1], 1, -1, -1), groups=p.shape[1])
        mu_t = torch.nn.functional.conv2d(t_pad, kernel.expand(t.shape[1], 1, -1, -1), groups=t.shape[1])
        
        # Variances
        var_p = torch.nn.functional.conv2d(p_pad ** 2, kernel.expand(p.shape[1], 1, -1, -1), groups=p.shape[1]) - mu_p ** 2
        var_t = torch.nn.functional.conv2d(t_pad ** 2, kernel.expand(t.shape[1], 1, -1, -1), groups=t.shape[1]) - mu_t ** 2
        cov_pt = torch.nn.functional.conv2d(p_pad * t_pad, kernel.expand(p.shape[1], 1, -1, -1), groups=p.shape[1]) - mu_p * mu_t
        
        # SSIM
        num = (2 * mu_p * mu_t + C1) * (2 * cov_pt + C2)
        den = (mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2)
        ssim_map = num / (den + 1e-8)
        
        return ssim_map.mean(dim=(-2, -1))


def load_manifest(path, root):
    """Load ROI manifest."""
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_manifest(manifest)
    names = sorted(p.name for p in (Path(root) / 'train' / 'DAPI').glob('*.jpg'))
    if digest(names) != manifest['inventory_names_sha256']:
        raise ValueError('Dataset inventory changed since split generation')
    return manifest


def make_loader(root, names, batch_size, augment=False, cache=True):
    """Create DataLoader."""
    ds = CleanPairedMarkers(root, names, augment=augment, cache=cache)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(SEMIFINAL_SEED)
    )


def transform(x, rotation, flip):
    """Apply geometric transformation."""
    x = torch.rot90(x, rotation, (-2, -1))
    return x.flip(-1) if flip else x


def inverse_transform(x, rotation, flip):
    """Inverse geometric transformation."""
    x = x.flip(-1) if flip else x
    return torch.rot90(x, -rotation, (-2, -1))


@torch.no_grad()
def predict(model, x, tta=1):
    """Predict with TTA."""
    variants = {
        1: [(0, False)],
        4: [(0, False), (0, True), (2, False), (2, True)],
        8: [(k, f) for k in range(4) for f in (False, True)]
    }
    if tta not in variants:
        raise ValueError('TTA must be 1, 4 or 8')
    
    result = None
    for k, f in variants[tta]:
        out = model(transform(x, k, f))
        if isinstance(out, tuple):
            out = out[0]
        out = inverse_transform(out.float(), k, f)
        result = out if result is None else result + out
    return result / len(variants[tta])


def evaluate(model, loader, device, tta=1, marker_names=MARKERS):
    """Evaluate model."""
    model.eval()
    rows = []
    start = time.perf_counter()
    
    marker_names = tuple(marker_names)
    
    for x, y, names in loader:
        x = x.to(device)
        y = y.to(device)
        
        # Expand DAPI to 3 channels if needed
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        
        with amp_context(device):
            p = predict(model, x, tta)
        
        # Compute SSIM and PSNR
        ssims = local_ssim(p.clamp(0, 1), y).cpu().numpy()
        mse = ((p - y) ** 2).mean(dim=(-2, -1)).cpu().numpy()
        psnrs = -10 * np.log10(np.maximum(mse, 1e-12))
        
        for name, ss, ps in zip(names, ssims, psnrs):
            rows.append({
                'name': name,
                'ssim': ss.tolist() if hasattr(ss, 'tolist') else float(ss),
                'psnr': ps.tolist() if hasattr(ps, 'tolist') else float(ps)
            })
    
    # Aggregate by marker
    marker_metrics = {}
    for i, m in enumerate(marker_names):
        ssims = [r['ssim'][i] if isinstance(r['ssim'], (list, np.ndarray)) else r['ssim'] for r in rows]
        psnrs = [r['psnr'][i] if isinstance(r['psnr'], (list, np.ndarray)) else r['psnr'] for r in rows]
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


def train(args):
    """Main training function."""
    print("=" * 60)
    print("Enhanced UNet++ Training")
    print("=" * 60)
    
    seed_all(args.seed)
    manifest = load_manifest(args.manifest, args.data_root)
    device = torch.device(args.device)
    
    # Get splits
    train_names = selected_names(manifest['splits']['train'], args.train_limit, args.seed)
    val_names = selected_names(manifest['splits']['val'], args.val_limit, args.seed)
    
    print(f"Train: {len(train_names)}, Val: {len(val_names)}")
    
    train_loader = make_loader(args.data_root, train_names, args.batch_size, augment=True)
    val_loader = make_loader(args.data_root, val_names, args.batch_size, augment=False) if val_names else None
    
    # Build model
    model = EnhancedUNetPlusPlus(
        in_channels=3,
        out_channels=len(MARKERS),
        base_channels=args.base_channels,
        depth=args.depth,
        use_deep_supervision=True,
        use_edge_loss=True,
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    
    def lr_lambda(step):
        warmup = args.warmup_epochs * len(train_loader)
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # EMA
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    ema_decay = 0.999
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        'model': 'EnhancedUNetPlusPlus',
        'base_channels': args.base_channels,
        'depth': args.depth,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'markers': list(MARKERS),
        'manifest_sha256': manifest['sha256'],
        'train_names': train_names,
        'val_names': val_names,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    best_ssim = -math.inf
    steps = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.perf_counter()
        
        for batch_idx, (dapi, ihc, _) in enumerate(train_loader):
            # Move to device
            dapi = dapi.to(device)
            ihc = ihc.to(device)
            
            # Expand DAPI to 3 channels
            if dapi.shape[1] == 1:
                dapi = dapi.expand(-1, 3, -1, -1)
            
            optimizer.zero_grad(set_to_none=True)
            
            with amp_context(device):
                # Forward
                out, deepsup, edge = model(dapi, return_all=True)
                
                # Compute edge target
                edge_target = torch.zeros_like(edge)
                
                # Combined loss
                loss, losses = combined_loss(
                    out, ihc,
                    edge_pred=edge,
                    edge_target=edge_target,
                    deepsup_preds=deepsup,
                    lambda_msssim=1.0,
                    lambda_l1=0.5,
                    lambda_edge=0.1,
                    lambda_color=0.05,
                    lambda_deepsup=0.2,
                )
            
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at epoch={epoch+1}, batch={batch_idx+1}")
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Update EMA
            with torch.no_grad():
                for ema_p, p in zip(ema.parameters(), model.parameters()):
                    ema_p.lerp_(p, 1 - ema_decay)
            
            steps += 1
            epoch_loss += loss.item()
            epoch_steps += 1
            
            if (batch_idx + 1) % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"LR: {lr:.2e}")
        
        # Evaluate
        if val_loader and (epoch + 1) % args.eval_every == 0:
            metrics = evaluate(ema, val_loader, device, tta=args.tta)
            print(f"\nValidation @ Epoch {epoch+1}:")
            print(f"  SSIM: {metrics['ssim']:.4f}")
            print(f"  PSNR: {metrics['psnr']:.2f}")
            for m, v in metrics['markers'].items():
                print(f"  {m}: SSIM={v['ssim']:.4f}, PSNR={v['psnr']:.2f}")
            
            if metrics['ssim'] > best_ssim:
                best_ssim = metrics['ssim']
                torch.save({
                    'epoch': epoch,
                    'steps': steps,
                    'model': model.state_dict(),
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_ssim': best_ssim,
                    'metrics': metrics,
                }, output_dir / 'best.pt')
                print(f"  -> New best model saved!")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'steps': steps,
            'model': model.state_dict(),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
        }, output_dir / 'last.pt')
        
        elapsed = time.perf_counter() - t0
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\nEpoch {epoch+1} completed in {elapsed:.1f}s, Loss: {avg_loss:.4f}\n")
    
    print(f"Training completed. Best SSIM: {best_ssim:.4f}")
    return best_ssim


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced UNet++ IHC model')
    
    parser.add_argument('--data-root', required=True, help='Data root directory')
    parser.add_argument('--manifest', required=True, help='ROI manifest JSON')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=SEMIFINAL_SEED, help='Random seed')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--base-channels', type=int, default=64, help='Base channels')
    parser.add_argument('--depth', type=int, default=4, help='Network depth')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Warmup epochs')
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--tta', type=int, default=4, choices=[1, 4, 8], help='TTA level')
    parser.add_argument('--train-limit', type=int, default=0, help='Limit training samples')
    parser.add_argument('--val-limit', type=int, default=0, help='Limit validation samples')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
