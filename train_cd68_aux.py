"""Fine-tune from w=96_expanded using CD68-aux architecture.

Strategy:
- Replace backbone with CD68AuxNet (adds CD68-aux head)
- Initialize backbone from w=96_expanded's EMA
- New CD68-aux block + head: randomly initialized
- Train with combined main + aux loss
"""
import sys, time, json, copy, math, random, hashlib
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.roi_manifest import MARKERS, SEMIFINAL_SEED, selected_names, validate_manifest
from src.models.marker_context_aux import CD68AuxNet, reconstruction_loss_with_cd68_aux
from src.models.marker_context import local_ssim
from src.train_marker_context import (make_loader, predict, amp_context, seed_all,
                                       write_json, environment)


def evaluate_per_marker(model, loader, device, tta=4):
    """Returns dict of {marker: {'ssim': ..., 'psnr': ...}, 'avg': {...}}"""
    model.eval()
    ssim_per = {m: [] for m in MARKERS}
    psnr_per = {m: [] for m in MARKERS}
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            # Use predict() which does TTA but returns main output only
            if tta > 1:
                pred = predict(model, x, tta=tta).float()
            else:
                pred = model.predict(x)
            for i, m in enumerate(MARKERS):
                p_i, y_i = pred[:, i:i+1], y[:, i:i+1]
                with torch.autocast(device_type=device.type, enabled=False):
                    s = local_ssim(p_i.float(), y_i.float()).mean().item()
                mse = ((p_i - y_i) ** 2).mean().item()
                psnr = 10 * np.log10(1.0 / max(mse, 1e-10)) if mse > 0 else 100.0
                ssim_per[m].append(s); psnr_per[m].append(psnr)
    out = {}
    for m in MARKERS:
        out[m] = {'ssim': float(np.mean(ssim_per[m])), 'psnr': float(np.mean(psnr_per[m]))}
    avg_s = float(np.mean([out[m]['ssim'] for m in MARKERS]))
    avg_p = float(np.mean([out[m]['psnr'] for m in MARKERS]))
    out['avg'] = {'ssim': avg_s, 'psnr': avg_p, 'score': avg_s * 100 + 0.128 * avg_p - 12.85}
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--init-checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--aux-weight', type=float, default=0.5)
    p.add_argument('--cd68-main-weight', type=float, default=1.0,
                    help='Extra weight on CD68 main head (in addition to aux).')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    seed_all(SEMIFINAL_SEED)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    validate_manifest(manifest)
    train_names = selected_names(manifest['splits']['train'], 0, SEMIFINAL_SEED)
    val_names = selected_names(manifest['splits']['val'], 0, SEMIFINAL_SEED)
    print(f"train={len(train_names)} val={len(val_names)}", flush=True)
    if val_names:
        print(f"holdout ROIs: {sorted(set(n.split('_')[0] for n in val_names))}", flush=True)

    # Load source checkpoint
    src = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
    src_run = src['run']
    src_config = src_run['model_config']
    src_width = src_config['width']
    print(f"Init from: width={src_width} (src epoch {src['epoch']})", flush=True)

    if src_run['split_sha256'] != manifest['sha256']:
        raise ValueError("split_sha mismatch")

    # Build CD68AuxNet with same width
    model = CD68AuxNet(width=src_width, markers=4, context=src_config.get('context', True)).to(device)

    # === CRITICAL: load compatible weights from src, leave aux parts random ===
    # src is a MarkerContextNet (single heads, no aux). We need to:
    # 1. Copy backbone weights (stem, encoders, down, context, up, fuse, decoders)
    # 2. Copy heads weights
    # 3. Leave cd68_aux_block and cd68_aux_head randomly initialized
    src_sd = src['ema']
    own_sd = model.state_dict()

    # Map: most keys are the same name. Skip cd68_aux_* keys.
    skipped_aux = 0
    for k, v in src_sd.items():
        if k.startswith('cd68_aux_'):
            skipped_aux += 1
            continue
        if k in own_sd:
            if own_sd[k].shape == v.shape:
                own_sd[k] = v
            else:
                print(f"  shape mismatch {k}: {own_sd[k].shape} vs {v.shape}", flush=True)
        else:
            print(f"  key not in target: {k}", flush=True)
    model.load_state_dict(own_sd, strict=True)
    print(f"Loaded backbone + main heads, skipped {skipped_aux} aux keys", flush=True)

    ema = copy.deepcopy(model).eval().requires_grad_(False)
    # For EMA, also load from src
    ema_sd = ema.state_dict()
    for k, v in src_sd.items():
        if k.startswith('cd68_aux_'):
            continue
        if k in ema_sd and ema_sd[k].shape == v.shape:
            ema_sd[k] = v
    ema.load_state_dict(ema_sd, strict=True)

    # Optimizer
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not torch.cuda.is_bf16_supported())

    train_loader = make_loader(args.data_root, train_names, args.batch_size, augment=True, cache=True)
    val_loader = make_loader(args.data_root, val_names, args.batch_size, cache=True) if val_names else None

    run = {
        'args': vars(args), 'model_config': {'width': src_width, 'markers': 4, 'context': True,
                                              'aux': 'cd68'},
        'markers': list(MARKERS), 'split_sha256': manifest['sha256'],
        'train_names': train_names, 'val_names': val_names,
        'init_checkpoint_sha256': hashlib.sha256(Path(args.init_checkpoint).read_bytes()).hexdigest(),
        'environment': environment(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'protocol': 'CD68-aux fine-tune from w96_expanded'
    }
    write_json(output / 'run.json', run)
    write_json(output / 'split.json', manifest)

    best, steps = -math.inf, 0
    for epoch in range(args.epochs):
        train_loader.generator.manual_seed(SEMIFINAL_SEED + epoch)
        random.seed(SEMIFINAL_SEED + epoch)
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
                pred_main, pred_aux = model(x)
            # Combined loss
            loss = reconstruction_loss_with_cd68_aux(pred_main, pred_aux, y,
                                                      aux_weight=args.aux_weight)
            # Add extra CD68 main weight
            if args.cd68_main_weight != 1.0:
                p_cd68 = pred_main[:, 1:2].float()
                t_cd68 = y[:, 1:2].float()
                cd68_extra = (1 - local_ssim(p_cd68, t_cd68).mean()
                              + 0.5 * F.l1_loss(p_cd68, t_cd68))
                loss = loss + (args.cd68_main_weight - 1.0) * cd68_extra
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

        # Validate
        metrics = evaluate_per_marker(ema, val_loader, device, tta=4) if val_loader else None
        if metrics:
            avg_ssim = metrics['avg']['ssim']
            improved = avg_ssim > best
            best = max(best, avg_ssim)
            record = {'epoch': epoch + 1, 'loss': total / max(count, 1), 'lr': lr,
                      'best_ssim': best, 'elapsed': time.perf_counter() - t0,
                      'validation': metrics}
            print(f"epoch={epoch+1} loss={total/max(count,1):.5f} "
                  f"CD68_SSIM={metrics['CD68']['ssim']:.4f} "
                  f"avg_SSIM={avg_ssim:.4f} "
                  f"avg_PSNR={metrics['avg']['psnr']:.3f} "
                  f"score={metrics['avg']['score']:.4f}", flush=True)
        else:
            record = {'epoch': epoch + 1, 'loss': total / max(count, 1), 'lr': lr,
                      'elapsed': time.perf_counter() - t0}
            print(f"epoch={epoch+1} loss={total/max(count,1):.5f}", flush=True)

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
        if metrics and improved:
            torch.save(state, output / 'best.tmp')
            (output / 'best.tmp').replace(output / 'best.pt')
            write_json(output / 'best_validation.json', metrics)
            print(f"  -> New best avg SSIM: {avg_ssim:.4f}", flush=True)
    print(f"Done. Best avg SSIM: {best:.4f}", flush=True)


if __name__ == '__main__':
    main()
