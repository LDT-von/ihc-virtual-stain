"""Bold training script: resume + flexible marker weights."""
import sys, time, json, copy, math, random, hashlib, io
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
    p, t = pred.float(), target.float()
    w = weights.to(p.device, dtype=p.dtype)
    ssim_per = local_ssim(p, t)
    ssim_loss = ((1 - ssim_per) * w).sum(dim=1).mean()
    l1_per = (p - t).abs().mean(dim=(-2, -1))
    l1 = (l1_per * w).sum(dim=1).mean()
    mse_per = ((p - t) ** 2).mean(dim=(-2, -1))
    mse = (mse_per * w).sum(dim=1).mean()
    loss = ssim_loss + 0.5 * l1 + 2 * mse
    for scale in (2, 4):
        ps = F.avg_pool2d(p, scale)
        ts = F.avg_pool2d(t, scale)
        l1_s_per = (ps - ts).abs().mean(dim=(-2, -1))
        l1_s = (l1_s_per * w).sum(dim=1).mean()
        loss = loss + 0.1 * l1_s
    return loss

def save_state(model, ema, optimizer, scaler, run, epoch, steps, output):
    state = {
        'format': 1, 'run': run, 'epoch': epoch, 'steps': steps,
        'model': model.state_dict(), 'ema': ema.state_dict(),
        'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
        'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                 'torch': torch.get_rng_state(),
                 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    }
    for name in ('last', 'final'):
        torch.save(state, output / f'{name}.tmp')
        (output / f'{name}.tmp').replace(output / f'{name}.pt')

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--init-checkpoint')  # Optional: initialize from checkpoint
    p.add_argument('--resume')           # Optional: resume from last.pt
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--cd68-weight', type=float, default=3.0)
    p.add_argument('--vimentin-weight', type=float, default=1.0)
    p.add_argument('--hla-weight', type=float, default=1.0)
    p.add_argument('--cd45ro-weight', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=SEMIFINAL_SEED)
    args = p.parse_args()

    seed_all(args.seed)
    device = torch.device('cuda')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    train_names = sorted(sum(manifest['splits'].values(), []))
    print(f"Total train (full-data): {len(train_names)}", flush=True)

    # Build model
    if args.resume:
        # Resume from last.pt
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_config = ckpt['run']['model_config']
        run = ckpt['run']
        run['args']['epochs'] = args.epochs  # extend epochs
        model = MarkerContextNet(**model_config).to(device)
        model.load_state_dict(ckpt['model'])
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        ema.load_state_dict(ckpt['ema'])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                       lr=args.lr, weight_decay=1e-4)
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        steps = ckpt['steps']
        random.setstate(ckpt['rng']['python'])
        np.random.set_state(ckpt['rng']['numpy'])
        torch.set_rng_state(ckpt['rng']['torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(ckpt['rng']['cuda'])
        print(f"RESUMED from epoch {ckpt['epoch']+1}, steps={steps}", flush=True)
    elif args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
        model_config = ckpt['run']['model_config']
        model = MarkerContextNet(**model_config).to(device)
        model.load_state_dict(ckpt['ema'], strict=True)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                       lr=args.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        start_epoch = 0
        steps = 0
        run = {
            'args': vars(args), 'model_config': model_config, 'markers': list(MARKERS),
            'split_sha256': manifest['sha256'], 'train_names': train_names, 'val_names': [],
            'marker_weights': dict(zip(MARKERS, [
                args.hla_weight, args.cd68_weight, args.cd45ro_weight, args.vimentin_weight])),
            'init_checkpoint_sha256': hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest(),
            'environment': environment(),
            'parameters': sum(p.numel() for p in model.parameters()),
            'protocol': 'bold continued refit with flexible marker weights',
        }
        write_json(output / 'run.json', run)
        write_json(output / 'split.json', manifest)
        print(f"INIT from {args.init_checkpoint}", flush=True)
    else:
        raise ValueError("Must provide --init-checkpoint or --resume")

    weights = torch.tensor([args.hla_weight, args.cd68_weight, args.cd45ro_weight, args.vimentin_weight])
    print(f"Marker weights: HLA={args.hla_weight} CD68={args.cd68_weight} "
          f"CD45RO={args.cd45ro_weight} Vim={args.vimentin_weight}", flush=True)
    print(f"LR={args.lr}, epochs={args.epochs}, batch={args.batch_size}", flush=True)

    total_epochs = start_epoch + args.epochs  # for cosine schedule
    if args.resume and start_epoch > 0:
        # Check if epochs match original
        orig_epochs = ckpt['run']['args'].get('epochs', 0)
        if orig_epochs and orig_epochs != args.epochs:
            print(f"WARNING: resuming with epochs={args.epochs} but original was {orig_epochs}", flush=True)
            print(f"  Cosine LR schedule may differ. EMA weights are correct.", flush=True)
    train_loader = make_loader(args.data_root, train_names, args.batch_size, augment=True, cache=True)

    for epoch in range(start_epoch, total_epochs):
        train_loader.generator.manual_seed(args.seed + epoch)
        random.seed(args.seed + epoch)
        model.train()
        t0, total, count = time.perf_counter(), 0., 0

        for batch_index, (x, y, _) in enumerate(train_loader):
            progress = (epoch + batch_index / len(train_loader)) / total_epochs
            lr = args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
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
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if not math.isfinite(grad_norm):
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

        save_state(model, ema, optimizer, scaler, run, epoch, steps, output)

    print(f"Done. Final model saved at {output/'final.pt'}", flush=True)

if __name__ == '__main__':
    main()
