"""Per-marker single-U-Net training with proper SSIM + multi-scale L1 loss.

Usage:
    # Train all 4 markers
    python -m src.train_marker_unet --marker HLA-DR --epochs 60 --batch-size 16
    python -m src.train_marker_unet --marker CD68 --epochs 60 --batch-size 16
    python -m src.train_marker_unet --marker CD45RO --epochs 60 --batch-size 16
    python -m src.train_marker_unet --marker Vimentin --epochs 60 --batch-size 16

    # Or use the automated runner
    python scripts/train_all_markers_unet.py
"""
import argparse
import copy
import hashlib
import io
import json
import math
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data.roi_manifest import MARKERS, build_manifest, digest, read_gray, roi_id, selected_names, validate_manifest
from .models.marker_losses import reconstruction_loss, local_ssim_4d
from .models.marker_unet import MarkerUNet, DeeperMarkerUNet


def amp_dtype(device):
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def isometric_augment(arr):
    """Apply random rotation + flip to all channels together (preserves alignment)."""
    arr = np.rot90(arr, random.randrange(4), axes=(-2, -1))
    if random.random() < 0.5:
        arr = arr[..., ::-1]
    return arr


def tta_predict(model, x, tta=4):
    """Test-time augmentation: average predictions over geometric transforms."""
    variants = {1: [(0, False)], 4: [(0, False), (0, True), (2, False), (2, True)],
                8: [(k, f) for k in range(4) for f in (False, True)]}
    if tta not in variants:
        raise ValueError('TTA must be 1, 4 or 8')
    
    def transform(t, k, f):
        t = torch.rot90(t, k, (-2, -1))
        return t.flip(-1) if f else t
    
    def inverse(t, k, f):
        t = t.flip(-1) if f else t
        return torch.rot90(t, -k, (-2, -1))
    
    results = []
    for k, f in variants[tta]:
        out = inverse(model(transform(x, k, f)), k, f).float()
        results.append(out)
    
    return torch.stack(results).mean(0).clamp(0, 1)


def jpeg_roundtrip(pred, quality):
    """Save to JPEG and reload to simulate submission compression."""
    images = pred.detach().cpu().mul(255).round().clamp(0, 255).byte().numpy()
    for image in images:
        from PIL import Image
        buffer = io.BytesIO()
        Image.fromarray(image[0]).save(buffer, format='JPEG', quality=quality, subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as im:
            image[:] = np.asarray(im)
    return torch.from_numpy(images.copy()).to(pred.device).float() / 255


class SingleMarkerDataset(torch.utils.data.Dataset):
    """DAPI + single marker paired dataset."""
    
    def __init__(self, root, names, marker, augment=True, cache=True):
        self.root = Path(root)
        self.names = list(names)
        self.marker = marker
        self.augment = augment
        
        if cache:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(4) as pool:
                self.cache = list(pool.map(self._read, self.names))
        else:
            self.cache = None
    
    def _read(self, name):
        dapi = read_gray(self.root / 'train' / 'DAPI' / name)
        target = read_gray(self.root / 'train' / self.marker / name)
        if dapi.shape != target.shape:
            raise ValueError(f'Shape mismatch for {name}: {dapi.shape} vs {target.shape}')
        return np.stack([dapi, target]).astype(np.float32)
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        arr = self.cache[idx] if self.cache is not None else self._read(self.names[idx])
        
        if self.augment:
            arr = isometric_augment(arr)
        
        dapi = torch.from_numpy(arr[0:1] / 255.0)
        target = torch.from_numpy(arr[1:2] / 255.0)
        return dapi, target, self.names[idx]


class TestDataset(torch.utils.data.Dataset):
    """Test set: DAPI only."""
    
    def __init__(self, dapi_dir):
        self.files = sorted(Path(dapi_dir).glob('*.jpg'))
        if not self.files:
            raise ValueError(f'No images in {dapi_dir}')
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        f = self.files[idx]
        dapi = read_gray(f)
        x = torch.from_numpy(dapi.astype(np.float32) / 255.0).unsqueeze(0)
        return x, f.stem


def local_ssim_np(pred, target):
    """Compute SSIM using numpy (for evaluation without torch autocast issues)."""
    from PIL import Image
    import numpy as np
    
    # Use skimage if available
    try:
        from skimage.metrics import structural_similarity as ssim
        p = np.asarray(pred * 255, dtype=np.uint8)
        t = np.asarray(target * 255, dtype=np.uint8)
        return ssim(p, t, data_range=255, channel_axis=None, use_sample_covariance=False)
    except ImportError:
        # Fallback: simple correlation-based
        p = pred.flatten()
        t = target.flatten()
        return np.corrcoef(p, t)[0, 1] if np.std(p) > 0 and np.std(t) > 0 else 0.0


def evaluate_model(model, loader, device, tta=1, use_jpeg=False, jpeg_quality=95):
    """Evaluate model on validation set."""
    model.eval()
    ssim_scores, psnr_scores, counts = [], [], 0
    
    with torch.no_grad():
        for x, y, names in loader:
            x, y = x.to(device), y.to(device)
            
            with torch.autocast(device_type=device.type, dtype=amp_dtype(device)):
                pred = tta_predict(model, x, tta)
            
            if use_jpeg:
                pred = jpeg_roundtrip(pred, jpeg_quality)
            
            # Compute per-image SSIM and PSNR
            for i in range(x.shape[0]):
                p = pred[i, 0].cpu().float()
                t = y[i, 0].cpu().float()
                ssim_val = local_ssim_np(p.numpy(), t.numpy())
                mse = ((p - t) ** 2).mean().item()
                psnr_val = -10 * np.log10(max(mse, 1e-12))
                ssim_scores.append(ssim_val)
                psnr_scores.append(psnr_val)
            
            counts += x.shape[0]
    
    return {
        'ssim': float(np.mean(ssim_scores)),
        'psnr': float(np.mean(psnr_scores)),
        'count': counts,
        'ssim_std': float(np.std(ssim_scores)),
    }


def make_loader(root, names, marker, batch_size, shuffle=True, cache=True):
    ds = SingleMarkerDataset(root, names, marker, augment=shuffle, cache=cache)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, 
                     num_workers=0, pin_memory=torch.cuda.is_available())


def build_model(model_type, device):
    """Build model by type."""
    if model_type == 'light':
        model = MarkerUNet(base_ch=64, residual_weight=0.0, use_attention=True)
    elif model_type == 'standard':
        model = MarkerUNet(base_ch=96, residual_weight=0.0, use_attention=True)
    elif model_type == 'deep':
        model = DeeperMarkerUNet(base_ch=48, dropout=0.1, residual_weight=0.0, use_attention=True)
    elif model_type == 'deep_large':
        model = DeeperMarkerUNet(base_ch=64, dropout=0.1, residual_weight=0.0, use_attention=True)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    return model.to(device)


def train_single_marker(args):
    """Train a single marker model."""
    seed_all(args.seed)
    
    # Load manifest
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    validate_manifest(manifest)
    
    # Verify data inventory
    names = sorted(p.name for p in (Path(args.data_root) / 'train' / 'DAPI').glob('*.jpg'))
    if digest(names) != manifest['inventory_names_sha256']:
        raise ValueError('Dataset inventory changed since split generation')
    
    marker = args.marker
    if marker not in MARKERS:
        raise ValueError(f'Unknown marker: {marker}. Choose from {MARKERS}')
    
    # Get splits
    train_names = selected_names(manifest['splits']['train'], args.train_limit, args.seed)
    val_names = selected_names(manifest['splits']['val'], args.val_limit, args.seed)
    
    print(f'[{marker}] Train: {len(train_names)}, Val: {len(val_names)}', flush=True)
    
    # Output directory
    run_dir = Path(args.output)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f'Output dir exists and non-empty: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sources
    sources_dir = run_dir / 'sources'
    sources_dir.mkdir(exist_ok=True)
    for src in [Path(__file__), Path(__file__).parent / 'models' / 'marker_unet.py',
                Path(__file__).parent / 'models' / 'marker_losses.py']:
        shutil.copy2(src, sources_dir / src.name)
    
    device = torch.device(args.device)
    model = build_model(args.model_type, device)
    
    print(f'[{marker}] Parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    
    # EMA for stable validation
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    ema_decay = 0.999
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # Dataloaders
    train_loader = make_loader(args.data_root, train_names, marker, args.batch_size, shuffle=True)
    val_loader = make_loader(args.data_root, val_names, marker, args.batch_size, shuffle=False)
    
    # Run metadata
    run_info = {
        'marker': marker,
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'train_count': len(train_names),
        'val_count': len(val_names),
        'parameters': sum(p.numel() for p in model.parameters()),
        'split_sha256': manifest['sha256'],
    }
    Path(run_dir / 'run.json').write_text(json.dumps(run_info, indent=2), encoding='utf-8')

    best_ssim = -1
    best_epoch = 0

    for epoch in range(args.epochs):
        # Shuffle augmentation per epoch
        train_loader.dataset.augment = True
        random.seed(args.seed + epoch)
        if hasattr(train_loader.dataset, 'cache') and train_loader.dataset.cache:
            random.shuffle(train_loader.dataset.cache)
        
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        t0 = time.perf_counter()
        
        for batch_idx, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device.type, dtype=amp_dtype(device)):
                pred = model(x)
                loss = reconstruction_loss(pred, y)
            
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch={epoch}, batch={batch_idx}')
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if not torch.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
                continue
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            with torch.no_grad():
                for ema_p, p in zip(ema.parameters(), model.parameters()):
                    ema_p.lerp_(p, 1 - ema_decay)
            
            epoch_loss += loss.item() * x.shape[0]
            epoch_count += x.shape[0]
        
        scheduler.step()
        avg_loss = epoch_loss / max(epoch_count, 1)
        elapsed = time.perf_counter() - t0
        
        # Evaluate with EMA
        metrics = evaluate_model(ema, val_loader, device, tta=args.tta)
        improved = metrics['ssim'] > best_ssim
        if improved:
            best_ssim = metrics['ssim']
            best_epoch = epoch + 1
        
        record = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'val_ssim': metrics['ssim'],
            'val_psnr': metrics['psnr'],
            'val_ssim_std': metrics['ssim_std'],
            'best_ssim': best_ssim,
            'best_epoch': best_epoch,
            'improved': improved,
            'elapsed': elapsed,
        }
        
        print(f'[{marker}] epoch={epoch+1}/{args.epochs} loss={avg_loss:.5f} '
              f'val_ssim={metrics["ssim"]:.4f} val_psnr={metrics["psnr"]:.2f} '
              f'best={best_ssim:.4f}@{best_epoch} [{elapsed:.1f}s]', flush=True)
        
        # Save checkpoint
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_ssim': best_ssim,
            'best_epoch': best_epoch,
            'val_ssim': metrics['ssim'],
            'val_psnr': metrics['psnr'],
            'run': run_info,  # for inference
        }
        torch.save(state, run_dir / 'last.pt')
        if improved:
            torch.save(state, run_dir / 'best.pt')
            Path(run_dir / 'best_validation.json').write_text(
                json.dumps(metrics, indent=2), encoding='utf-8')
        
        # Append to history
        with (run_dir / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    
    print(f'[{marker}] Done. Best SSIM={best_ssim:.4f} at epoch {best_epoch}', flush=True)
    return best_ssim, best_epoch


def infer_single_marker(args):
    """Generate predictions for test set using a trained model."""
    seed_all(args.seed)
    device = torch.device(args.device)
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Build model
    model = build_model(ckpt['run']['model_type'], device)
    model.load_state_dict(ckpt['ema'])
    model.eval()
    
    marker = ckpt['run']['marker']
    print(f'[{marker}] Loading from {args.checkpoint}, val_ssim={ckpt.get("val_ssim", "N/A")}', flush=True)
    
    # Test dataset
    test_ds = TestDataset(Path(args.input) / 'DAPI')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)
    
    output = Path(args.output)
    marker_dir = output / 'results' / 'test' / marker
    marker_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    t0 = time.perf_counter()
    
    with torch.no_grad():
        for x, names in test_loader:
            x = x.to(device)
            
            with torch.autocast(device_type=device.type, dtype=amp_dtype(device)):
                pred = tta_predict(model, x, args.tta)
            
            # Save as JPEG
            for i in range(x.shape[0]):
                img = pred[i, 0].cpu().mul(255).round().clamp(0, 255).byte().numpy()
                from PIL import Image
                Image.fromarray(img).save(
                    marker_dir / f'{names[i]}_fake.jpg',
                    quality=args.jpeg_quality, subsampling=0
                )
            
            count += x.shape[0]
            if count % 200 == 0:
                print(f'[{marker}] {count}/{len(test_ds)}', flush=True)
    
    elapsed = time.perf_counter() - t0
    print(f'[{marker}] Done. Wrote {count} images in {elapsed:.1f}s', flush=True)
    
    # Provenance
    Path(output / 'provenance.json').write_text(json.dumps({
        'marker': marker,
        'count': count,
        'tta': args.tta,
        'jpeg_quality': args.jpeg_quality,
        'checkpoint': args.checkpoint,
    }, indent=2), encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    
    # Split command
    sp = sub.add_parser('split')
    sp.add_argument('--data-root', required=True)
    sp.add_argument('--output', required=True)
    sp.add_argument('--seed', type=int, default=42)
    sp.add_argument('--val-rois', type=int, default=3)
    sp.add_argument('--holdout-rois', type=int, default=3)
    
    # Train command
    tp = sub.add_parser('train')
    tp.add_argument('--marker', required=True, choices=list(MARKERS))
    tp.add_argument('--data-root', required=True)
    tp.add_argument('--manifest', required=True)
    tp.add_argument('--output', required=True)
    tp.add_argument('--model-type', default='standard', 
                   choices=['light', 'standard', 'deep', 'deep_large'])
    tp.add_argument('--epochs', type=int, default=60)
    tp.add_argument('--batch-size', type=int, default=16)
    tp.add_argument('--lr', type=float, default=5e-4)
    tp.add_argument('--device', default='cuda')
    tp.add_argument('--seed', type=int, default=42)
    tp.add_argument('--tta', type=int, default=4, choices=[1, 4, 8])
    tp.add_argument('--train-limit', type=int, default=0)
    tp.add_argument('--val-limit', type=int, default=0)
    tp.add_argument('--resume', action='store_true')
    
    # Infer command
    ip = sub.add_parser('infer')
    ip.add_argument('--checkpoint', required=True)
    ip.add_argument('--input', required=True)
    ip.add_argument('--output', required=True)
    ip.add_argument('--batch-size', type=int, default=32)
    ip.add_argument('--device', default='cuda')
    ip.add_argument('--seed', type=int, default=42)
    ip.add_argument('--tta', type=int, default=4, choices=[1, 4, 8])
    ip.add_argument('--jpeg-quality', type=int, default=100)
    
    args = parser.parse_args()
    
    if args.command == 'split':
        manifest = build_manifest(args.data_root, args.output, args.seed, 
                                 args.val_rois, args.holdout_rois)
        print(json.dumps({k: {'count': len(v), 'rois': sorted({roi_id(n) for n in v})}
                          for k, v in manifest['splits'].items()}))
    
    elif args.command == 'train':
        train_single_marker(args)
    
    elif args.command == 'infer':
        infer_single_marker(args)


if __name__ == '__main__':
    main()
