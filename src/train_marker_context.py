"""Auditable ROI-disjoint multi-marker training, evaluation and submission inference."""
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
from torch.utils.data import DataLoader

from .data.roi_manifest import (MARKERS, PairedMarkers, build_manifest, digest,
                                read_gray, roi_id, selected_names, validate_manifest)
from .models.marker_context import MarkerContextNet, local_ssim, reconstruction_loss


def amp_context(device):
    # Ampere supports BF16's FP32-like exponent range, avoiding FP16 gradient overflow.
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
    # Reverse the operation order: undo flip first, then rotation.
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
    return torch.from_numpy(images.copy()).to(pred.device).float()/255


@torch.no_grad()
def evaluate(model, loader, device, tta=1, jpeg_quality=0):
    model.eval()
    rows = []
    start = time.perf_counter()
    for x, y, names in loader:
        x, y = x.to(device), y.to(device)
        with amp_context(device):
            p = predict(model, x, tta)
        p = jpeg_roundtrip(p, jpeg_quality) if jpeg_quality else (p*255).round()/255
        scores = local_ssim(p, y).cpu().numpy()
        errors = (p-y).square().mean((-2, -1)).cpu().numpy()
        psnrs = -10*np.log10(np.maximum(errors, 1e-12))
        for name, ss, ps in zip(names, scores, psnrs):
            rows.append({'name': name, 'roi': roi_id(name), 'ssim': ss.tolist(), 'psnr': ps.tolist()})
    if not rows:
        raise ValueError('Empty evaluation set')
    marker = {m: {'ssim': float(np.mean([r['ssim'][i] for r in rows])),
                  'psnr': float(np.mean([r['psnr'][i] for r in rows]))} for i, m in enumerate(MARKERS)}
    per_roi = {r: {k: float(np.mean([row[k] for row in rows if row['roi'] == r])) for k in ('ssim','psnr')}
               for r in sorted({row['roi'] for row in rows})}
    return {'count': len(rows), 'ssim': float(np.mean([m['ssim'] for m in marker.values()])),
            'psnr': float(np.mean([m['psnr'] for m in marker.values()])), 'markers': marker,
            'rois': per_roi, 'tta': tta, 'jpeg_quality': jpeg_quality,
            'seconds': time.perf_counter()-start, 'rows': rows}


def compact(metrics):
    return {k: v for k, v in metrics.items() if k != 'rows'}


def load_manifest(path, root):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_manifest(manifest)
    names = sorted(p.name for p in (Path(root)/'train'/'DAPI').glob('*.jpg'))
    if digest(names) != manifest['inventory_names_sha256']:
        raise ValueError('Dataset inventory changed since split generation')
    return manifest


def environment():
    project_root = Path(__file__).resolve().parents[1]
    try:
        git = (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project_root,
                stderr=subprocess.DEVNULL, timeout=5, text=True).strip()
               if (project_root/'.git').exists() else 'unavailable (source bundle without .git)')
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        git = 'unavailable'
    sources = [Path(__file__), Path(__file__).parent/'models'/'marker_context.py',
               Path(__file__).parent/'data'/'roi_manifest.py']
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
    full_data = getattr(args, 'full_data', False)
    init_checkpoint = getattr(args, 'init_checkpoint', None)
    if args.resume and init_checkpoint:
        raise ValueError('--resume restores a run; --init-checkpoint starts a new run. Choose one.')
    if full_data and not (init_checkpoint or args.resume):
        raise ValueError('Full-data refit requires a validated --init-checkpoint or --resume')
    if full_data and (args.train_limit or args.val_limit):
        raise ValueError('Full-data refit cannot use subset limits')
    output = Path(args.output)
    if args.resume and Path(args.resume).resolve().parent != output.resolve():
        raise ValueError('--resume must use the original output directory; use --init-checkpoint for a new run')
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Use a fresh run directory or explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output/'sources'
    source_dir.mkdir(exist_ok=True)
    for source in (Path(__file__), Path(__file__).parent/'models'/'marker_context.py',
                   Path(__file__).parent/'data'/'roi_manifest.py'):
        if not (source_dir/source.name).exists():
            shutil.copy2(source, source_dir/source.name)
    train_names = (sorted(sum(manifest['splits'].values(), [])) if full_data else
                   selected_names(manifest['splits']['train'], args.train_limit, args.seed))
    val_names = [] if full_data else selected_names(manifest['splits']['val'], args.val_limit, args.seed)
    model_config = dict(width=args.width, markers=len(MARKERS), context=not args.no_context)
    protocol = 'all-data refit; NO held-out metric' if full_data else 'ROI disjoint development'
    print(f'Loading train={len(train_names)} val={len(val_names)}; {protocol}', flush=True)
    train_loader = make_loader(args.data_root, train_names, args.batch_size, True, not args.no_cache)
    val_loader = make_loader(args.data_root, val_names, args.batch_size, cache=not args.no_cache) if val_names else None
    model = MarkerContextNet(**model_config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not torch.cuda.is_bf16_supported())
    start, steps, best = 0, 0, -math.inf
    run = {'args': vars(args), 'model_config': model_config, 'markers': list(MARKERS),
           'split_sha256': manifest['sha256'], 'train_names': train_names, 'val_names': val_names,
           'environment': environment(), 'parameters': sum(p.numel() for p in model.parameters()),
           'protocol': protocol,
           'selection': 'fixed final epoch; no validation' if full_data else
                        'maximum validation mean SSIM; PSNR reported separately; not an official composite score'}
    if init_checkpoint:
        initial = torch.load(init_checkpoint, map_location='cpu', weights_only=False)
        if initial['run']['model_config'] != model_config or initial['run']['split_sha256'] != manifest['sha256']:
            raise ValueError('Initialization model/split mismatch')
        if tuple(initial['run']['markers']) != MARKERS:
            raise ValueError('Initialization marker order mismatch')
        if not full_data and set(initial['run']['train_names'])-set(manifest['splits']['train']):
            raise ValueError('Initialization already saw validation/holdout ROIs')
        model.load_state_dict(initial['ema'], strict=True)
        ema.load_state_dict(initial['ema'], strict=True)
        run['initial_checkpoint_sha256'] = hashlib.sha256(Path(init_checkpoint).read_bytes()).hexdigest()
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        for key in ('model_config', 'split_sha256', 'train_names', 'val_names'):
            if ckpt['run'][key] != run[key]:
                raise ValueError(f'Resume mismatch: {key}')
        # Epoch count controls the cosine schedule; extend by a separate fine-tuning run.
        for key in ('epochs', 'lr', 'batch_size', 'seed'):
            if ckpt['run']['args'][key] != getattr(args, key):
                raise ValueError(f'Resume must preserve {key}')
        if ckpt['run']['environment']['source_sha256'] != run['environment']['source_sha256']:
            raise ValueError('Training source changed; use --init-checkpoint for an explicit new experiment')
        model.load_state_dict(ckpt['model'], strict=True)
        ema.load_state_dict(ckpt['ema'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start, steps, best = ckpt['epoch']+1, ckpt['steps'], ckpt['best_ssim']
        random.setstate(ckpt['rng']['python'])
        np.random.set_state(ckpt['rng']['numpy'])
        torch.set_rng_state(ckpt['rng']['torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(ckpt['rng']['cuda'])
    write_json(output/'run.json', run)
    write_json(output/'split.json', manifest)
    print(f'parameters={run["parameters"]:,} device={device} output={output}', flush=True)
    for epoch in range(start, args.epochs):
        # Match sample order and isometric augmentation across architecture ablations.
        train_loader.generator.manual_seed(args.seed+epoch)
        random.seed(args.seed+epoch)
        model.train()
        t0, total, count, skipped = time.perf_counter(), 0., 0, 0
        for batch_index, (x, y, _) in enumerate(train_loader):
            progress = (epoch + batch_index/len(train_loader))/args.epochs
            lr = args.lr*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                pred = model(x)
            loss = reconstruction_loss(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss epoch={epoch} step={batch_index}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled())
            if not torch.isfinite(grad_norm):
                # GradScaler skips this optimizer update and reduces its scale.
                scaler.step(optimizer)
                scaler.update()
                skipped += 1
                if skipped > 10:
                    raise FloatingPointError('Repeated FP16 overflow; use a BF16-capable GPU or CPU')
                continue
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            decay = min(.995, (1+steps)/(10+steps))
            with torch.no_grad():
                for e, p in zip(ema.parameters(), model.parameters()):
                    e.lerp_(p, 1-decay)
            total += loss.item()*len(x)
            count += len(x)
            if (batch_index+1) % 100 == 0:
                print(f'epoch={epoch+1} batch={batch_index+1}/{len(train_loader)} loss={total/count:.5f}', flush=True)
        metrics = evaluate(ema, val_loader, device) if val_loader is not None else None
        improved = metrics is not None and metrics['ssim'] > best
        if metrics is not None:
            best = max(best, metrics['ssim'])
        record = {'epoch': epoch+1, 'loss': total/max(count, 1), 'lr': lr, 'best_ssim': None if full_data else best,
                  'elapsed': time.perf_counter()-t0, 'skipped_overflow_batches': skipped,
                  'validation': compact(metrics) if metrics is not None else None}
        print(json.dumps(record), flush=True)
        with (output/'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        state = {'format': 1, 'run': run, 'epoch': epoch, 'steps': steps, 'best_ssim': best,
                 'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scaler': scaler.state_dict(), 'metrics': metrics,
                 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                         'torch': torch.get_rng_state(),
                         'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}
        torch.save(state, output/'last.tmp')
        (output/'last.tmp').replace(output/'last.pt')
        if improved:
            torch.save(state, output/'best.tmp')
            (output/'best.tmp').replace(output/'best.pt')
            write_json(output/'best_validation.json', metrics)
    if full_data:
        shutil.copy2(output/'last.pt', output/'final.pt')
        print('Done; full-data final.pt is NOT eligible for held-out evaluation', flush=True)
    else:
        print(f'Done; best validation SSIM={best:.6f}; official score NOT measured', flush=True)


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != 1 or tuple(checkpoint['run']['markers']) != MARKERS:
        raise ValueError('Unsupported checkpoint/marker order')
    model = MarkerContextNet(**checkpoint['run']['model_config']).to(device)
    model.load_state_dict(checkpoint['ema'], strict=True)
    return model.eval(), checkpoint


def eval_command(args):
    seed_all(args.seed)
    manifest = load_manifest(args.manifest, args.data_root)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    if checkpoint['run']['split_sha256'] != manifest['sha256']:
        raise ValueError('Checkpoint/split mismatch')
    names = selected_names(manifest['splits'][args.split], args.limit, args.seed)
    # Independent checks prevent evaluation mislabeled as held-out after full-data training.
    if set(names) & set(checkpoint['run']['train_names']):
        raise ValueError('Evaluation names were used for training')
    loader = make_loader(args.data_root, names, args.batch_size)
    metrics = evaluate(model, loader, device, args.tta, args.jpeg_quality)
    metrics.update({'split': args.split, 'split_sha256': manifest['sha256'],
                    'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, metrics)
    print(json.dumps(compact(metrics)), flush=True)


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
        raise FileExistsError('Inference output must be empty to avoid stale mixed predictions')
    for marker in MARKERS:
        (output/'results'/'test'/marker).mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(inputs), args.batch_size):
        files = inputs[offset:offset+args.batch_size]
        arrays = [read_gray(f) for f in files]
        if any(a.shape != (256, 256) for a in arrays):
            raise ValueError('Expected 256x256 official inputs')
        x = torch.from_numpy(np.stack(arrays)[:, None]).float().to(device)/255
        with amp_context(device):
            pred = predict(model, x, args.tta).mul(255).round().byte().cpu().numpy()
        for file, channels in zip(files, pred):
            for marker, channel in zip(MARKERS, channels):
                Image.fromarray(channel).save(output/'results'/'test'/marker/(file.stem+'_fake.jpg'),
                                              quality=args.jpeg_quality, subsampling=0)
        if offset % (args.batch_size*50) == 0:
            print(f'Inferred {min(offset+len(files), len(inputs))}/{len(inputs)}', flush=True)
    write_json(output/'provenance.json', {'input_count': len(inputs), 'markers': list(MARKERS),
               'tta': args.tta, 'jpeg_quality': args.jpeg_quality, 'run': checkpoint['run'],
               'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
               'input_names_sha256': digest([p.name for p in inputs])})
    print(f'Wrote {len(inputs)*len(MARKERS)} JPEGs. Package only results/, not provenance.json.', flush=True)


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
            p.add_argument('--width', type=int, default=16)
            p.add_argument('--epochs', type=int, default=20)
            p.add_argument('--lr', type=float, default=5e-4)
            p.add_argument('--no-context', action='store_true')
            p.add_argument('--no-cache', action='store_true')
            p.add_argument('--train-limit', type=int, default=0)
            p.add_argument('--val-limit', type=int, default=0)
            p.add_argument('--resume')
            p.add_argument('--init-checkpoint', help='Initialize EMA weights only; start a fresh optimizer/schedule')
            p.add_argument('--full-data', action='store_true', help='Refit all labeled ROIs after locking choices; disables validation')
        else:
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--tta', type=int, choices=(1, 4, 8), default=1)
            p.add_argument('--jpeg-quality', type=int, default=95)
            if name == 'eval':
                p.add_argument('--split', choices=('val', 'holdout'), default='val')
                p.add_argument('--limit', type=int, default=0)
            else:
                p.add_argument('--input', required=True)
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
