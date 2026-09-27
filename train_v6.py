"""v6: 300 epoch single-cosine with progressive augmentation weakening.

  - Scratch init (no v3/v4/v5 init)
  - 300 epoch cosine LR with 10-epoch warmup
  - save_every 50 (snapshots: e50, e100, ..., e300 + final)
  - Augmentation strength weakens linearly from 1.0 -> 0.5 over the last 100 epochs
    to give the model a clean final fit
  - CD68 weight 3.0 unchanged
"""
import sys, time, json, copy, math, random, hashlib
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.data.roi_manifest import MARKERS, SEMIFINAL_SEED
from src.models.marker_context import MarkerContextNet, local_ssim
from src.train_marker_context import (make_loader, amp_context, seed_all,
                                       write_json, environment)


def weighted_loss(pred, target, weights):
    """Per-marker weighted loss with multi-scale L1 supervision."""
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
    p.add_argument('--output', required=True)
    p.add_argument('--init-checkpoint', default=None)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--cd68-weight', type=float, default=3.0)
    p.add_argument('--warmup-epochs', type=int, default=10)
    p.add_argument('--weaken-start-epoch', type=int, default=200,
                   help='From this epoch onward aug_strength ramps 1.0 -> 0.5 by --epochs')
    p.add_argument('--seed', type=int, default=SEMIFINAL_SEED)
    p.add_argument('--width', type=int, default=96)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device('cuda')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    train_names = sorted(sum(manifest['splits'].values(), []))
    print(f"Total train: {len(train_names)}", flush=True)

    weights = torch.tensor([1.0, args.cd68_weight, 1.0, 1.0])
    print(f"Marker weights: {dict(zip(MARKERS, weights.tolist()))}", flush=True)

    if args.init_checkpoint:
        src = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        src_run = src['run']
        model_config = src_run['model_config']
        init_kind = f"init={Path(args.init_checkpoint).name}"
    else:
        model_config = {'width': args.width, 'markers': 4, 'context': True}
        init_kind = 'init=scratch'
    print(f"Model config: {model_config}  ({init_kind})", flush=True)
    model = MarkerContextNet(**model_config).to(device)
    if args.init_checkpoint:
        model.load_state_dict(src['ema'], strict=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not torch.cuda.is_bf16_supported())

    run = {
        'args': vars(args), 'model_config': model_config, 'markers': list(MARKERS),
        'split_sha256': manifest['sha256'], 'train_names': train_names, 'val_names': [],
        'marker_weights': dict(zip(MARKERS, weights.tolist())),
        'init_checkpoint_sha256': hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest() if args.init_checkpoint else None,
        'environment': environment(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'protocol': f'v6 300ep single-cosine + progressive aug weaken (init={init_kind}, cd68w={args.cd68_weight})',
    }
    write_json(output / 'run.json', run)
    write_json(output / 'split.json', manifest)

    def lr_at(epoch, batch_progress):
        if epoch < args.warmup_epochs:
            return args.lr * (epoch + batch_progress) / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs + batch_progress) / max(args.epochs - args.warmup_epochs, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    def aug_strength_at(epoch):
        if epoch < args.weaken_start_epoch:
            return 1.0
        t = (epoch - args.weaken_start_epoch) / max(args.epochs - args.weaken_start_epoch, 1)
        return max(0.5, 1.0 - 0.5 * t)

    # Build the loader with the starting strength; we'll rebuild for weaken window.
    current_strength = aug_strength_at(0)
    train_loader = make_loader(args.data_root, train_names, args.batch_size,
                               augment=True, cache=True, aug_strength=current_strength)

    def snapshot(epoch, steps):
        return {'format': 1, 'run': run, 'epoch': epoch, 'steps': steps,
                'model': model.state_dict(), 'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}

    steps = 0
    for epoch in range(args.epochs):
        # Update aug strength at epoch boundary
        new_strength = aug_strength_at(epoch)
        if abs(new_strength - current_strength) > 0.01:
            current_strength = new_strength
            train_loader = make_loader(args.data_root, train_names, args.batch_size,
                                       augment=True, cache=True, aug_strength=current_strength)
            print(f"  >> rebuild loader with aug_strength={current_strength:.2f}", flush=True)

        train_loader.generator.manual_seed(args.seed + epoch)
        random.seed(args.seed + epoch)
        model.train()
        t0, total, count = time.perf_counter(), 0., 0
        for batch_index, (x, y, _) in enumerate(train_loader):
            batch_progress = batch_index / len(train_loader)
            lr = lr_at(epoch, batch_progress)
            for group in optimizer.param_groups:
                group['lr'] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                pred = model(x)
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
                  'aug_strength': current_strength,
                  'elapsed': time.perf_counter() - t0}
        print(f"epoch={epoch+1} loss={total/max(count,1):.5f} aug_str={current_strength:.2f} elapsed={record['elapsed']:.1f}s", flush=True)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

        state = snapshot(epoch, steps)
        torch.save(state, output / 'last.tmp')
        (output / 'last.tmp').replace(output / 'last.pt')
        torch.save(state, output / 'final.tmp')
        (output / 'final.tmp').replace(output / 'final.pt')

        if args.save_every and (epoch + 1) % args.save_every == 0:
            sp = output / f'snapshot_epoch_{epoch+1:03d}.pt'
            torch.save(state, sp)
            print(f"saved snapshot: {sp.name}", flush=True)

    print(f"Done. Final at {output/'final.pt'}, snapshots at snapshot_epoch_*.pt", flush=True)


if __name__ == '__main__':
    main()
