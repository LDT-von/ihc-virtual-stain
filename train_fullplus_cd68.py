"""Fine-tune w=96_expanded on fullplus (all 2380 ROIs) for 5 epochs with CD68 weighting.
This is a 'no-holdout' final-stage refit, mimicking how platform distribution might look.
"""
import sys, time, json, copy, math, random, hashlib, io
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from src.data.roi_manifest import MARKERS, SEMIFINAL_SEED, validate_manifest
from src.models.marker_context import MarkerContextNet, local_ssim, reconstruction_loss
from src.train_marker_context import (make_loader, amp_context, seed_all,
                                       write_json, environment, OFFICIAL_JPEG_QUALITY)


def jpeg_roundtrip_train_train(pred, quality=OFFICIAL_JPEG_QUALITY):
    """Apply official JPEG roundtrip to predictions during training.
    Uses in-place numpy operations to avoid gradient issues."""
    # Work on CPU with uint8 conversion (gradient-preserving)
    images = pred.detach().cpu().mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
    result = np.empty_like(images)
    for i, img in enumerate(images):
        for c in range(img.shape[0]):
            buf = io.BytesIO()
            Image.fromarray(img[c]).save(buf, format='JPEG', quality=quality,
                                         subsampling=0, optimize=False)
            buf.seek(0)
            with Image.open(buf) as im:
                result[i, c] = np.asarray(im)
    return torch.from_numpy(result.copy()).to(pred.device, dtype=pred.dtype)/255.0


def weighted_loss(pred, target, weights):
    p, t = pred.float(), target.float()
    weights = weights.to(p.device, dtype=p.dtype)
    ssim_per = local_ssim(p, t)
    ssim_loss = ((1 - ssim_per) * weights).sum(dim=1).mean()
    l1_per = (p - t).abs().mean(dim=(-2, -1))
    l1 = (l1_per * weights).sum(dim=1).mean()
    mse_per = ((p - t) ** 2).mean(dim=(-2, -1))
    mse = (mse_per * weights).sum(dim=1).mean()
    loss = ssim_loss + 0.5 * l1 + 2 * mse
    for scale in (2, 4):
        ps = F.avg_pool2d(p, scale)
        ts = F.avg_pool2d(t, scale)
        l1_s_per = (ps - ts).abs().mean(dim=(-2, -1))
        l1_s = (l1_s_per * weights).sum(dim=1).mean()
        loss = loss + 0.1 * l1_s
    return loss


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--init-checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--cd68-weight', type=float, default=3.0)
    p.add_argument('--seed', type=int, default=SEMIFINAL_SEED)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device('cuda')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    train_names = sorted(sum(manifest['splits'].values(), []))
    print(f"Total train (full-data from manifest splits): {len(train_names)}", flush=True)
    print(f"Source ROIs: {sorted(set(n.split('_')[0] for n in train_names))}", flush=True)
    print(f"Manifest inventory_sha: {manifest['inventory_names_sha256'][:16]}", flush=True)
    print(f"Manifest sha256: {manifest['sha256'][:16]}", flush=True)
    # NOTE: skip validate_manifest - we use expanded's validated manifest and take all splits

    src = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
    src_run = src['run']
    if src_run['split_sha256'] != manifest['sha256']:
        print(f"WARNING: src split {src_run['split_sha256'][:8]} != manifest {manifest['sha256'][:8]}", flush=True)
        print(f"  src train_names ⊆ manifest train+val: "
              f"{set(src_run['train_names']).issubset(set(train_names))}", flush=True)
    
    # Build model with src config
    model_config = src_run['model_config']
    model = MarkerContextNet(**model_config).to(device)
    model.load_state_dict(src['ema'], strict=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not torch.cuda.is_bf16_supported())
    
    weights = torch.tensor([1.0, args.cd68_weight, 1.0, 1.0])
    print(f"Marker weights: {dict(zip(MARKERS, weights.tolist()))}", flush=True)
    
    train_loader = make_loader(args.data_root, train_names, args.batch_size, augment=True, cache=True)
    
    run = {
        'args': vars(args), 'model_config': model_config, 'markers': list(MARKERS),
        'split_sha256': manifest['sha256'], 'train_names': train_names, 'val_names': [],
        'marker_weights': dict(zip(MARKERS, weights.tolist())),
        'init_checkpoint_sha256': hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest(),
        'environment': environment(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'protocol': 'fullplus CD68-weighted refit (all 2380 ROIs) + official augment + JPEG roundtrip',
    }
    write_json(output / 'run.json', run)
    write_json(output / 'split.json', manifest)
    
    steps = 0
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
            # NOTE: Skip JPEG roundtrip during training - it's non-differentiable.
            # The model learns with continuous predictions; JPEG is only applied at inference.
            loss = weighted_loss(pred, y, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss epoch={epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled())
            if not torch.isfinite(grad_norm):
                scaler.step(optimizer); scaler.update(); continue
            scaler.step(optimizer); scaler.update()
            steps += 1
            decay = min(.995, (1 + steps) / (10 + steps))
            with torch.no_grad():
                for e, p in zip(ema.parameters(), model.parameters()):
                    if p.requires_grad:
                        e.lerp_(p, 1 - decay)
            total += loss.item() * len(x); count += len(x)
            if (batch_index + 1) % 100 == 0:
                print(f"epoch={epoch+1} batch={batch_index+1}/{len(train_loader)} "
                      f"loss={total/count:.5f} lr={lr:.6f}", flush=True)
        
        record = {'epoch': epoch + 1, 'loss': total / max(count, 1), 'lr': lr,
                  'elapsed': time.perf_counter() - t0}
        print(f"epoch={epoch+1} loss={total/max(count,1):.5f} elapsed={record['elapsed']:.1f}s", flush=True)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        
        # Save last as final (no validation, no 'best')
        state = {'format': 1, 'run': run, 'epoch': epoch, 'steps': steps,
                 'model': model.state_dict(), 'ema': ema.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                         'torch': torch.get_rng_state(),
                         'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}
        torch.save(state, output / 'last.tmp')
        (output / 'last.tmp').replace(output / 'last.pt')
        # Also save as final
        torch.save(state, output / 'final.tmp')
        (output / 'final.tmp').replace(output / 'final.pt')
    print(f"Done. Final model saved at {output/'final.pt'}", flush=True)


if __name__ == '__main__':
    main()
