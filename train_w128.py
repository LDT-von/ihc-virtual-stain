"""Train w=128 from scratch on expanded data.
Hypothesis: larger backbone capacity benefits CD68 (largest SSIM gap).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time, json, copy, math, random, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data.roi_manifest import MARKERS, PairedMarkers, SEMIFINAL_SEED, selected_names, validate_manifest
from src.models.marker_context import MarkerContextNet, local_ssim, reconstruction_loss
from src.train_marker_context import (build_reconstruction_model, make_loader, predict,
                                       amp_context, seed_all, write_json, evaluate,
                                       environment, transform, inverse_transform)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--train-limit', type=int, default=0)
    p.add_argument('--val-limit', type=int, default=0)
    p.add_argument('--seed', type=int, default=SEMIFINAL_SEED)
    args = p.parse_args()
    
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    validate_manifest(manifest)
    train_names = selected_names(manifest['splits']['train'], args.train_limit, args.seed)
    val_names = selected_names(manifest['splits']['val'], args.val_limit, args.seed)
    print(f"train={len(train_names)} val={len(val_names)}", flush=True)
    if val_names:
        val_rois = sorted(set(n.split('_')[0] for n in val_names))
        print(f"holdout ROIs: {val_rois}", flush=True)
    
    model_config = dict(width=args.width, markers=len(MARKERS), context=True)
    model = build_reconstruction_model(model_config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not torch.cuda.is_bf16_supported())
    
    train_loader = make_loader(args.data_root, train_names, args.batch_size, augment=True, cache=True)
    val_loader = make_loader(args.data_root, val_names, args.batch_size, cache=True) if val_names else None
    
    run = {
        'args': vars(args), 'model_config': model_config, 'markers': list(MARKERS),
        'split_sha256': manifest['sha256'], 'train_names': train_names, 'val_names': val_names,
        'environment': environment(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'protocol': 'w=128 fresh training from expanded manifest',
    }
    write_json(output / 'run.json', run)
    write_json(output / 'split.json', manifest)
    
    best, steps = -math.inf, 0
    for epoch in range(args.epochs):
        train_loader.generator.manual_seed(args.seed + epoch)
        random.seed(args.seed + epoch)
        model.train()
        t0, total, count = time.perf_counter(), 0., 0
        for batch_index, (x, y, _) in enumerate(train_loader):
            progress = (epoch + batch_index / len(train_loader)) / args.epochs
            lr = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                pred = model(x)
            loss = reconstruction_loss(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss epoch={epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled())
            if not torch.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
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
                print(f"epoch={epoch+1} batch={batch_index+1}/{len(train_loader)} "
                      f"loss={total/count:.5f}", flush=True)
        
        metrics = evaluate(ema, val_loader, device) if val_loader else None
        improved = metrics is not None and metrics['ssim'] > best
        if metrics is not None:
            best = max(best, metrics['ssim'])
        record = {'epoch': epoch + 1, 'loss': total / max(count, 1), 'lr': lr,
                  'best_ssim': best, 'elapsed': time.perf_counter() - t0,
                  'validation': metrics}
        print(json.dumps(record), flush=True)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        
        state = {'format': 1, 'run': run, 'epoch': epoch, 'steps': steps, 'best_ssim': best,
                 'model': model.state_dict(), 'ema': ema.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                 'metrics': metrics,
                 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                         'torch': torch.get_rng_state(),
                         'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}
        torch.save(state, output / 'last.tmp')
        (output / 'last.tmp').replace(output / 'last.pt')
        if improved:
            torch.save(state, output / 'best.tmp')
            (output / 'best.tmp').replace(output / 'best.pt')
            write_json(output / 'best_validation.json', metrics)
            print(f"  -> New best holdout SSIM: {metrics['ssim']:.4f}", flush=True)
    
    print(f"Done. Best holdout SSIM: {best:.4f}", flush=True)


if __name__ == '__main__':
    main()
