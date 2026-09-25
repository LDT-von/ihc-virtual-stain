"""CD68-weighted fine-tuning from w96_expanded baseline.

Hypothesis: CD68 SSIM is the largest gap. Apply per-marker loss weighting to
emphasize CD68 in gradient updates, while fine-tuning from existing checkpoint.
"""
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.roi_manifest import (MARKERS, PairedMarkers, SEMIFINAL_SEED,
                                    selected_names, validate_manifest)
from src.models.marker_context import (MarkerContextNet, local_ssim,
                                        reconstruction_loss)
from src.train_marker_context import (build_reconstruction_model,
                                       make_loader, predict, jpeg_roundtrip,
                                       environment, write_json, amp_context,
                                       seed_all)

# Per-marker loss weights: CD68 (index 1) gets higher weight to push SSIM
# CD68 index = MARKERS.index('CD68') = 1
MARKER_WEIGHTS = torch.tensor([1.0, 3.0, 1.0, 1.0])  # HLA-DR, CD68, CD45RO, Vimentin


def weighted_reconstruction_loss(pred, target, weights=None):
    """Per-marker weighted reconstruction loss.

    Applies a per-channel weight to the SSIM + L1 + MSE components of the
    standard reconstruction loss. This boosts gradients on markers (e.g. CD68)
    that have a higher per-marker weight.
    """
    if weights is None:
        return reconstruction_loss(pred, target)
    p, t = pred.float(), target.float()
    weights = weights.to(p.device, dtype=p.dtype)

    # Per-marker SSIM loss: 1 - SSIM, then weight
    # local_ssim returns (N, C) after reducing over spatial dims
    ssim_per = local_ssim(p, t)  # (N, C)
    ssim_loss_per = 1 - ssim_per
    ssim_loss = (ssim_loss_per * weights).sum(dim=1).mean()

    # Per-marker L1
    l1_per = (p - t).abs().mean(dim=(-2, -1))  # (N, C)
    l1 = (l1_per * weights).sum(dim=1).mean()

    # Per-marker MSE
    mse_per = ((p - t) ** 2).mean(dim=(-2, -1))  # (N, C)
    mse = (mse_per * weights).sum(dim=1).mean()

    loss = ssim_loss + 0.5 * l1 + 2 * mse
    # Multi-scale L1 supervision
    for scale in (2, 4):
        ps = F.avg_pool2d(p, scale)
        ts = F.avg_pool2d(t, scale)
        l1_s_per = (ps - ts).abs().mean(dim=(-2, -1))  # (N, C)
        l1_s = (l1_s_per * weights).sum(dim=1).mean()
        loss = loss + 0.1 * l1_s
    return loss


@torch.no_grad()
def evaluate_per_marker(model, loader, device, tta=4):
    model.eval()
    ssim_per = {m: [] for m in MARKERS}
    psnr_per = {m: [] for m in MARKERS}
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = predict(model, x, tta=tta).float()
        for i, m in enumerate(MARKERS):
            p_i = pred[:, i:i+1]
            y_i = y[:, i:i+1]
            with torch.autocast(device_type=device.type, enabled=False):
                s = local_ssim(p_i.float(), y_i.float()).mean().item()
            mse = ((p_i - y_i) ** 2).mean().item()
            psnr = 10 * np.log10(1.0 / max(mse, 1e-10)) if mse > 0 else 100.0
            ssim_per[m].append(s)
            psnr_per[m].append(psnr)
    return {m: {'ssim': float(np.mean(ssim_per[m])), 'psnr': float(np.mean(psnr_per[m]))}
            for m in MARKERS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--init-checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--cd68-weight', type=float, default=3.0)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    seed_all(SEMIFINAL_SEED)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Build marker weights: HLA-DR, CD68, CD45RO, Vimentin
    weights = torch.tensor([1.0, args.cd68_weight, 1.0, 1.0])
    print(f"Marker weights: {dict(zip(MARKERS, weights.tolist()))}", flush=True)

    # Load manifest
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    validate_manifest(manifest)
    train_names = selected_names(manifest['splits']['train'], 0, SEMIFINAL_SEED)
    val_names = selected_names(manifest['splits']['val'], 0, SEMIFINAL_SEED)
    print(f"train={len(train_names)} val={len(val_names)}", flush=True)
    if val_names:
        val_rois = sorted(set(n.split('_')[0] for n in val_names))
        print(f"holdout ROIs: {val_rois}", flush=True)

    # Load checkpoint
    ckpt = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
    run = ckpt['run']
    model_config = run['model_config']
    print(f"Init from: width={run['args']['width']} epochs={run['args']['epochs']}", flush=True)

    # Important: split must match
    if run['split_sha256'] != manifest['sha256']:
        raise ValueError(f"Split mismatch: ckpt={run['split_sha256']} vs manifest={manifest['sha256']}")
    if tuple(run['markers']) != MARKERS:
        raise ValueError("Marker order mismatch")

    # Build model and load EMA weights (not 'model' weights - EMA is better)
    model = build_reconstruction_model(model_config).to(device)
    model.load_state_dict(ckpt['ema'], strict=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    # Fresh optimizer and cosine schedule
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not torch.cuda.is_bf16_supported())

    # Loaders
    train_loader = make_loader(args.data_root, train_names, args.batch_size, augment=True, cache=True)
    val_loader = make_loader(args.data_root, val_names, args.batch_size, cache=True) if val_names else None

    run_meta = {
        'args': vars(args),
        'model_config': model_config,
        'markers': list(MARKERS),
        'split_sha256': manifest['sha256'],
        'train_names': train_names,
        'val_names': val_names,
        'marker_weights': dict(zip(MARKERS, weights.tolist())),
        'init_checkpoint_sha256': hashlib_path(args.init_checkpoint),
        'environment': environment(),
        'protocol': 'CD68-weighted fine-tuning from w96_expanded',
    }
    import hashlib
    write_json(output / 'run.json', run_meta)
    write_json(output / 'split.json', manifest)

    steps = 0
    best_cd68_ssim = -math.inf
    best_avg_ssim = -math.inf

    for epoch in range(args.epochs):
        train_loader.generator.manual_seed(SEMIFINAL_SEED + epoch)
        random.seed(SEMIFINAL_SEED + epoch)
        model.train()
        t0, total, count, skipped = time.perf_counter(), 0., 0, 0
        for batch_index, (x, y, _) in enumerate(train_loader):
            progress = (epoch + batch_index / len(train_loader)) / args.epochs
            lr = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                pred = model(x)
            loss = weighted_reconstruction_loss(pred, y, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss epoch={epoch} step={batch_index}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                  error_if_nonfinite=not scaler.is_enabled())
            if not torch.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
                skipped += 1
                continue
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            decay = min(.995, (1 + steps) / (10 + steps))
            with torch.no_grad():
                for e, p in zip(ema.parameters(), model.parameters()):
                    if p.requires_grad:
                        e.lerp_(p, 1 - decay)
            total += loss.item() * len(x)
            count += len(x)
            if (batch_index + 1) % 100 == 0:
                print(f"epoch={epoch + 1} batch={batch_index + 1}/{len(train_loader)} "
                      f"loss={total / count:.5f} lr={lr:.6f}", flush=True)

        # Per-marker validation
        metrics = evaluate_per_marker(ema, val_loader, device, tta=4) if val_loader else None
        if metrics:
            avg_ssim = float(np.mean([metrics[m]['ssim'] for m in MARKERS]))
            avg_psnr = float(np.mean([metrics[m]['psnr'] for m in MARKERS]))
            cd68_ssim = metrics['CD68']['ssim']
            improved_cd68 = cd68_ssim > best_cd68_ssim
            improved_avg = avg_ssim > best_avg_ssim
            best_cd68_ssim = max(best_cd68_ssim, cd68_ssim)
            best_avg_ssim = max(best_avg_ssim, avg_ssim)
            record = {
                'epoch': epoch + 1, 'loss': total / max(count, 1), 'lr': lr,
                'elapsed': time.perf_counter() - t0,
                'skipped_overflow_batches': skipped,
                'validation': metrics,
                'avg_ssim': avg_ssim, 'avg_psnr': avg_psnr,
                'best_cd68_ssim': best_cd68_ssim, 'best_avg_ssim': best_avg_ssim,
            }
            print(f"epoch={epoch + 1} loss={total / max(count, 1):.5f} "
                  f"CD68_SSIM={cd68_ssim:.4f} avg_SSIM={avg_ssim:.4f} "
                  f"avg_PSNR={avg_psnr:.3f}", flush=True)
        else:
            record = {'epoch': epoch + 1, 'loss': total / max(count, 1), 'lr': lr,
                      'elapsed': time.perf_counter() - t0,
                      'skipped_overflow_batches': skipped}

        with (output / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

        state = {'format': 1, 'run': run_meta, 'epoch': epoch, 'steps': steps,
                 'best_cd68_ssim': best_cd68_ssim, 'best_avg_ssim': best_avg_ssim,
                 'model': model.state_dict(), 'ema': ema.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                 'metrics': metrics,
                 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                         'torch': torch.get_rng_state(),
                         'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}

        torch.save(state, output / 'last.tmp')
        (output / 'last.tmp').replace(output / 'last.pt')
        if metrics and improved_cd68:
            torch.save(state, output / 'best.tmp')
            (output / 'best.tmp').replace(output / 'best.pt')
            write_json(output / 'best_cd68_validation.json', metrics)
            print(f"  -> New best CD68 SSIM: {cd68_ssim:.4f}", flush=True)

    print(f"Done. Best CD68 SSIM: {best_cd68_ssim:.4f}, Best Avg SSIM: {best_avg_ssim:.4f}", flush=True)


def hashlib_path(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    main()
