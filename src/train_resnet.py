"""ResNet50-UNet training, evaluation and inference.

Command examples:
    # Split
    python -m src.train_resnet split --data-root /data1/AIC/roi_dataset --output roi_split_resnet.json

    # Train
    python -m src.train_resnet train \
        --data-root /data1/AIC/roi_dataset \
        --manifest /data1/AIC/roi_dataset/roi_split_v1.json \
        --output checkpoints/resnet_unet_v1 \
        --epochs 60 --batch-size 16 --lr 1e-3 \
        --device cuda:1

    # Eval holdout
    python -m src.train_resnet eval \
        --data-root /data1/AIC/roi_dataset \
        --manifest /data1/AIC/roi_dataset/roi_split_v1.json \
        --checkpoint checkpoints/resnet_unet_v1/best.pt \
        --split holdout --tta 8 \
        --output eval_holdout.json --device cuda:1

    # Inference test
    python -m src.train_resnet infer \
        --checkpoint checkpoints/resnet_unet_v1/best.pt \
        --input /data1/AIC/roi_dataset/test/DAPI \
        --output preds/resnet_unet_v1 \
        --tta 8 --jpeg-quality 95 --device cuda:1
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
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data.roi_manifest import (MARKERS, PairedMarkers, build_manifest, digest,
                                read_gray, roi_id, selected_names, validate_manifest)
from .models.resnet_unet import ResNetUNet, reconstruction_loss as model_loss


def amp_context(device):
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def transform(x, rotation, flip):
    x = torch.rot90(x, rotation, (-2, -1))
    return x.flip(-1) if flip else x


def inverse_transform(x, rotation, flip):
    x = x.flip(-1) if flip else x
    return torch.rot90(x, -rotation, (-2, -1))


@torch.no_grad()
def predict(model, x, tta=1):
    variants = {1: [(0, False)], 4: [(0, False), (0, True), (2, False), (2, True)],
                8: [(k, f) for k in range(4) for f in (False, True)]}
    if tta not in variants:
        raise ValueError('TTA must be 1, 4 or 8')
    result = None
    for k, f in variants[tta]:
        # x is (B, C, H, W) where C = markers; rot90/flip are spatial only
        output = inverse_transform(model(transform(x, k, f)), k, f).float()
        result = output if result is None else result + output
    return result.div_(len(variants[tta])).clamp_(0, 1)


def jpeg_roundtrip(pred, quality):
    images = pred.detach().cpu().mul(255).round().clamp(0, 255).byte().numpy()
    for image in images:
        for channel in image:
            buffer = io.BytesIO()
            Image.fromarray(channel).save(buffer, format='JPEG', quality=quality, subsampling=0)
            buffer.seek(0)
            with Image.open(buffer) as im:
                channel[:] = np.asarray(im)
    return torch.from_numpy(images.copy()).to(pred.device).float() / 255


def local_ssim(pred, target):
    """Per-marker SSIM, spatial mean. Returns tensor of shape (B, markers)."""
    pred = pred.float()
    target = target.float()
    C = pred.shape[1]
    losses = []
    for c in range(C):
        p, t = pred[:, c:c+1], target[:, c:c+1]
        mu_x = F.avg_pool2d(p, 5, 1, 2)
        mu_y = F.avg_pool2d(t, 5, 1, 2)
        sigma_x = F.avg_pool2d(p.pow(2), 5, 1, 2) - mu_x.pow(2)
        sigma_y = F.avg_pool2d(t.pow(2), 5, 1, 2) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(p * t, 5, 1, 2) - mu_x * mu_y
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2))
        losses.append(ssim.mean(dim=(-2, -1)))
    return torch.stack(losses, dim=1)


@torch.no_grad()
def evaluate(model, loader, device, tta=1, jpeg_quality=0):
    model.eval()
    rows = []
    start = time.perf_counter()
    for x, y, names in loader:
        x, y = x.to(device), y.to(device)
        with amp_context(device):
            p = predict(model, x, tta)
        p = jpeg_roundtrip(p, jpeg_quality) if jpeg_quality else (p * 255).round() / 255
        scores = local_ssim(p, y).cpu().numpy()
        errors = (p - y).square().mean((-2, -1)).cpu().numpy()
        psnrs = -10 * np.log10(np.maximum(errors, 1e-12))
        for name, ss, ps in zip(names, scores, psnrs):
            rows.append({'name': name, 'roi': roi_id(name), 'ssim': ss.tolist(), 'psnr': ps.tolist()})
    if not rows:
        raise ValueError('Empty evaluation set')
    marker = {m: {'ssim': float(np.mean([r['ssim'][i] for r in rows])),
                  'psnr': float(np.mean([r['psnr'][i] for r in rows]))}
              for i, m in enumerate(MARKERS)}
    per_roi = {r: {k: float(np.mean([row[k] for row in rows if row['roi'] == r]))
                   for k in ('ssim', 'psnr')}
               for r in sorted({row['roi'] for row in rows})}
    return {'count': len(rows), 'ssim': float(np.mean([m['ssim'] for m in marker.values()])),
            'psnr': float(np.mean([m['psnr'] for m in marker.values()])), 'markers': marker,
            'rois': per_roi, 'tta': tta, 'jpeg_quality': jpeg_quality,
            'seconds': time.perf_counter() - start, 'rows': rows}


def compact(metrics):
    return {k: v for k, v in metrics.items() if k != 'rows'}


def load_manifest(path, root):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_manifest(manifest)
    names = sorted(p.name for p in (Path(root) / 'train' / 'DAPI').glob('*.jpg'))
    if digest(names) != manifest['inventory_names_sha256']:
        raise ValueError('Dataset inventory changed since split generation')
    return manifest


def environment():
    project_root = Path(__file__).resolve().parents[1]
    try:
        git = (subprocess.check_output(['git', 'rev-parse', '--HEAD'], cwd=project_root,
                stderr=subprocess.DEVNULL, timeout=5, text=True).strip()
               if (project_root / '.git').exists() else 'unavailable (source bundle without .git)')
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        git = 'unavailable'
    sources = [Path(__file__), Path(__file__).parent / 'models' / 'resnet_unet.py',
               Path(__file__).parent / 'data' / 'roi_manifest.py']
    return {'python': sys.version, 'torch': str(torch.__version__), 'platform': platform.platform(),
            'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
            'git_head': git, 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def make_loader(root, names, batch, augment=False, cache=True):
    ds = PairedMarkers(root, names, augment=augment, cache=cache)
    return DataLoader(ds, batch_size=batch, shuffle=augment, num_workers=0,
                      pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(42))


def train(args):
    seed_all(args.seed)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('epochs, batch size and learning rate must be positive')
    manifest = load_manifest(args.manifest, args.data_root)
    device = torch.device(args.device)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh run directory or explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / 'sources'
    source_dir.mkdir(exist_ok=True)
    for source in (Path(__file__), Path(__file__).parent / 'models' / 'resnet_unet.py',
                   Path(__file__).parent / 'data' / 'roi_manifest.py'):
        shutil.copy2(source, source_dir / source.name)

    train_names = selected_names(manifest['splits']['train'], args.train_limit, args.seed)
    val_names = selected_names(manifest['splits']['val'], args.val_limit, args.seed)
    print(f'Loading train={len(train_names)} val={len(val_names)}; ROI disjoint development', flush=True)

    train_loader = make_loader(args.data_root, train_names, args.batch_size, True, not args.no_cache)
    val_loader = make_loader(args.data_root, val_names, args.batch_size, cache=not args.no_cache) if val_names else None

    model = ResNetUNet(markers=len(MARKERS), pretrained=True).to(device)
    print(f'parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable', flush=True)

    # Decoder is trainable from scratch; encoder is fine-tuned (smaller LR).
    encoder_params = [p for n, p in model.named_parameters()
                      if any(k in n for k in ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'))]
    decoder_params = [p for n, p in model.named_parameters()
                      if not any(k in n for k in ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'))]
    print(f'encoder params: {sum(p.numel() for p in encoder_params):,}', flush=True)
    print(f'decoder params: {sum(p.numel() for p in decoder_params):,}', flush=True)
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.lr * 0.1},   # encoder fine-tune
        {'params': decoder_params, 'lr': args.lr},          # decoder train from scratch
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    ema = copy.deepcopy(model).eval().requires_grad_(False)

    run = {'args': vars(args), 'markers': list(MARKERS),
           'split_sha256': manifest['sha256'], 'train_names': train_names, 'val_names': val_names,
           'environment': environment(), 'parameters': sum(p.numel() for p in model.parameters()),
           'protocol': 'ROI disjoint development; ImageNet pretrained ResNet50 encoder',
           'selection': 'maximum validation mean SSIM'}

    write_json(output / 'run.json', run)
    write_json(output / 'split.json', manifest)
    print(f'output={output} device={device}', flush=True)

    best = -math.inf
    steps = 0
    for epoch in range(args.epochs):
        train_loader.generator.manual_seed(args.seed + epoch)
        random.seed(args.seed + epoch)
        model.train()
        t0, total, count = time.perf_counter(), 0.0, 0
        for batch_index, (x, y, _) in enumerate(train_loader):
            progress = (epoch + batch_index / len(train_loader)) / args.epochs
            lr = args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))
            for pg, lr_mult in zip(optimizer.param_groups, [0.1, 1.0]):
                pg['lr'] = lr * lr_mult

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                pred = model(x)
            loss = model_loss(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss epoch={epoch} step={batch_index}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            count += x.shape[0]

        elapsed = time.perf_counter() - t0
        train_loss = total / count
        print(f'epoch={epoch:3d} loss={train_loss:.4f} lr_enc={optimizer.param_groups[0]["lr"]:.2e} '
              f'lr_dec={optimizer.param_groups[1]["lr"]:.2e} [{elapsed:.0f}s]', flush=True)

        # EMA update — skip non-float tensors (BN running counters are int64)
        with torch.no_grad():
            for ema_p, model_p in zip(ema.state_dict().values(), model.state_dict().values()):
                if ema_p.is_floating_point():
                    # Use adaptive decay: fast catch-up early, slow late.
                    decay = min(0.999, (1 + steps) / (10 + steps))
                    ema_p.mul_(decay).add_(model_p.to(ema_p.dtype), alpha=1 - decay)
        steps += 1

        # Validation
        if val_loader is not None and ((epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1):
            metrics = evaluate(ema, val_loader, device, tta=8)
            improved = metrics['ssim'] > best
            if improved:
                best = metrics['ssim']
                improved_flag = '*'
            else:
                improved_flag = ''
            print(f'  val  ssim={metrics["ssim"]:.4f} psnr={metrics["psnr"]:.2f}{improved_flag} '
                  f'[{metrics["seconds"]:.1f}s]', flush=True)
            state = {'format': 1, 'run': run, 'model': model.state_dict(), 'ema': ema.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                     'epoch': epoch, 'best_ssim': best,
                     'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                             'torch': torch.get_rng_state(),
                             'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}}
            torch.save(state, output / 'last.pt')
            if improved:
                torch.save(state, output / 'best.tmp')
                (output / 'best.tmp').replace(output / 'best.pt')
                write_json(output / 'best_validation.json', metrics)
    print(f'Done; best validation SSIM={best:.6f}', flush=True)


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if tuple(checkpoint['run']['markers']) != MARKERS:
        raise ValueError('Marker order mismatch')
    model = ResNetUNet(markers=len(MARKERS), pretrained=False).to(device)
    model.load_state_dict(checkpoint['ema'], strict=True)
    return model.eval(), checkpoint


@torch.no_grad()
def eval_command(args):
    seed_all(args.seed)
    manifest = load_manifest(args.manifest, args.data_root)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    if checkpoint['run']['split_sha256'] != manifest['sha256']:
        raise ValueError('Checkpoint/split mismatch')
    names = selected_names(manifest['splits'][args.split], args.limit, args.seed)
    if set(names) & set(checkpoint['run']['train_names']):
        raise ValueError('Evaluation names were used for training')
    loader = make_loader(args.data_root, names, args.batch_size)
    metrics = evaluate(model, loader, device, args.tta, args.jpeg_quality)
    metrics.update({'split': args.split, 'split_sha256': manifest['sha256'],
                    'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, metrics)
    print(json.dumps(compact(metrics)), flush=True)


@torch.no_grad()
def infer_command(args):
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError('JPEG quality must be 1..100')
    seed_all(args.seed)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    inputs = sorted(Path(args.input).glob('*.jpg'))
    if not inputs:
        raise ValueError('No input JPEGs')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Inference output must be empty')
    for marker in MARKERS:
        (output / 'results' / 'test' / marker).mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(inputs), args.batch_size):
        files = inputs[offset:offset + args.batch_size]
        arrays = [read_gray(f) for f in files]
        if any(a.shape != (256, 256) for a in arrays):
            raise ValueError('Expected 256x256 official inputs')
        x = torch.from_numpy(np.stack(arrays)[:, None]).float().to(device) / 255
        with amp_context(device):
            pred = predict(model, x, args.tta).mul(255).round().byte().cpu().numpy()
        for file, channels in zip(files, pred):
            for marker, channel in zip(MARKERS, channels):
                Image.fromarray(channel).save(
                    output / 'results' / 'test' / marker / (file.stem + '_fake.jpg'),
                    quality=args.jpeg_quality, subsampling=0)
        if offset % (args.batch_size * 50) == 0:
            print(f'Inferred {min(offset + len(files), len(inputs))}/{len(inputs)}', flush=True)
    write_json(output / 'provenance.json', {
        'input_count': len(inputs), 'markers': list(MARKERS),
        'tta': args.tta, 'jpeg_quality': args.jpeg_quality, 'run': checkpoint['run'],
        'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'input_names_sha256': digest([p.name for p in inputs])})
    print(f'Wrote {len(inputs) * len(MARKERS)} JPEGs.', flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    s = sub.add_parser('split')
    s.add_argument('--data-root', required=True)
    s.add_argument('--output', required=True)
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--val-rois', type=int, default=3)
    s.add_argument('--holdout-rois', type=int, default=3)

    for name in ('train', 'eval', 'infer'):
        p = sub.add_parser(name)
        p.add_argument('--output', required=True)
        p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--batch-size', type=int, default=4)
        if name != 'infer':
            p.add_argument('--data-root', required=True)
            p.add_argument('--manifest', required=True)
        if name == 'train':
            p.add_argument('--epochs', type=int, default=60)
            p.add_argument('--lr', type=float, default=1e-3)
            p.add_argument('--train-limit', type=int, default=0)
            p.add_argument('--val-limit', type=int, default=0)
            p.add_argument('--val-every', type=int, default=5)
            p.add_argument('--no-cache', action='store_true')
        elif name == 'eval':
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--split', choices=('val', 'holdout'), default='holdout')
            p.add_argument('--tta', type=int, choices=(1, 4, 8), default=8)
            p.add_argument('--jpeg-quality', type=int, default=0)
            p.add_argument('--limit', type=int, default=0)
        else:
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--input', required=True)
            p.add_argument('--tta', type=int, choices=(1, 4, 8), default=8)
            p.add_argument('--jpeg-quality', type=int, default=95)

    args = parser.parse_args()
    if args.command == 'split':
        manifest = build_manifest(args.data_root, args.output, args.seed, args.val_rois, args.holdout_rois)
        print(json.dumps({k: {'count': len(v), 'rois': sorted({roi_id(n) for n in v})}
                          for k, v in manifest['splits'].items()}))
    elif args.command == 'train':
        train(args)
    elif args.command == 'eval':
        eval_command(args)
    else:
        infer_command(args)


if __name__ == '__main__':
    main()
