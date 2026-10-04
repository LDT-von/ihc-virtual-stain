"""ROI-held-out semifinal training with explicit augmentation and loss ablations.

This is separate from the older initial-round train_v7.py and from full-data
train_v6.py. Run ``fit`` for every ROI fold, ``summarize`` to lock one epoch,
then ``fit --stage final`` to refit on all labeled ROIs. Nothing runs on import.
"""

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.paired_clean import CleanPairedMarkers
from src.data.marker_subset import MarkerSubsetDataset
from src.data.roi_manifest import MARKERS, PairedMarkers, SEMIFINAL_SEED, roi_id
from src.models.marker_context import local_ssim
from src.models.marker_smp_unet import SMP_VERSION
from src.train_marker_context import (
    _NEW_CHANNEL_ARCHITECTURES,
    _WIDTH_IS_DECODER_CHANNELS,
    amp_context,
    build_reconstruction_model,
    compact,
    environment,
    evaluate,
    infer_command,
    load_manifest,
    seed_all,
    update_ema,
    write_json,
)


def reconstruction_loss(pred, target, weights, policy):
    """Keep the V6 terms; change only their reduction for normalized ablation."""
    p, t = pred.float(), target.float()
    w = weights.to(device=p.device, dtype=p.dtype)
    per_marker = 1 - local_ssim(p, t)
    per_marker = per_marker + 0.5 * (p - t).abs().mean(dim=(-2, -1))
    per_marker = per_marker + 2.0 * (p - t).square().mean(dim=(-2, -1))
    for scale in (2, 4):
        coarse_p = F.avg_pool2d(p, scale)
        coarse_t = F.avg_pool2d(t, scale)
        per_marker = per_marker + 0.1 * (coarse_p - coarse_t).abs().mean(dim=(-2, -1))
    weighted = (per_marker * w).sum(dim=1)
    if policy == 'normalized':
        weighted = weighted / w.sum()
    elif policy != 'legacy-v6':
        raise ValueError(f'Unsupported loss policy: {policy}')
    return weighted.mean()


def make_folds(names, folds, seed):
    rois = sorted({roi_id(name) for name in names})
    if folds < 2 or folds > len(rois):
        raise ValueError(f'folds must be in [2, {len(rois)}]')
    random.Random(seed).shuffle(rois)
    return [sorted(rois[i::folds]) for i in range(folds)]


def recipe_from_args(args):
    recipe = {
        'architecture': args.architecture,
        'target_marker': args.target_marker,
        'width': args.width,
        'augmentation': args.augmentation,
        'loss': args.loss,
        'cd68_weight': args.cd68_weight,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'schedule_epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'weaken_start_epoch': args.weaken_start_epoch,
        'eval_every': args.eval_every,
        'tta': args.tta,
        'folds': args.folds,
        'seed': args.seed,
    }
    if args.architecture == 'prototype_marker':
        recipe.update({
            'num_shared_prototypes': args.num_shared_prototypes,
            'num_task_prototypes': args.num_task_prototypes,
            'prototype_temperature': args.prototype_temperature,
            'prototype_diversity_weight': args.prototype_diversity_weight,
        })
    if args.architecture == 'smp_resnet34_unet':
        recipe['smp_version'] = SMP_VERSION
    return recipe


def model_config_from_args(args, marker_names):
    config = {'width': args.width, 'markers': len(marker_names)}
    if args.architecture not in _WIDTH_IS_DECODER_CHANNELS + _NEW_CHANNEL_ARCHITECTURES:
        config['context'] = True
    if args.architecture != 'context':
        config['architecture'] = args.architecture
    if args.architecture == 'smp_resnet34_unet':
        config['smp_version'] = SMP_VERSION
    if args.architecture == 'prototype_marker':
        config.update({
            'num_shared_prototypes': args.num_shared_prototypes,
            'num_task_prototypes': args.num_task_prototypes,
            'temperature': args.prototype_temperature,
        })
    return config


def make_training_loader(args, names, device, marker_names):
    if args.augmentation == 'legacy-v6':
        dataset = PairedMarkers(args.data_root, names, augment=True, cache=not args.no_cache)
    else:
        dataset = CleanPairedMarkers(
            args.data_root, names, augment=True, cache=not args.no_cache,
            policy=args.augmentation,
        )
    if tuple(marker_names) != MARKERS:
        dataset = MarkerSubsetDataset(dataset, marker_names)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == 'cuda', generator=generator,
    )
    return loader


def learning_rate(args, epoch, batch_fraction):
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + batch_fraction) / max(args.warmup_epochs, 1)
    progress = (epoch - args.warmup_epochs + batch_fraction) / max(
        args.epochs - args.warmup_epochs, 1
    )
    return args.lr * 0.5 * (1 + math.cos(math.pi * progress))


def augmentation_strength(args, epoch):
    if epoch < args.weaken_start_epoch:
        return 1.0
    progress = (epoch - args.weaken_start_epoch) / max(
        args.epochs - args.weaken_start_epoch, 1
    )
    return max(0.5, 1.0 - 0.5 * progress)


def source_hashes():
    root = Path(__file__).resolve().parent
    names = (
        'train_semifinal_v7.py',
        'src/data/paired_clean.py',
        'src/data/roi_manifest.py',
        'src/data/marker_subset.py',
        'src/models/marker_context.py',
        'src/models/marker_specific.py',
        'src/models/prototype_marker.py',
        'src/models/marker_nafnet.py',
        'src/models/marker_smp_unet.py',
        'src/models/enhanced_unetpp.py',
        'src/models/marigold_ihc.py',
        'src/models/transformer_ihc.py',
        'requirements-smp-unet.txt',
        'src/train_marker_context.py',
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names}


def repair_history(path, checkpoint):
    """Keep the history aligned with the last atomic checkpoint after an interruption."""
    history_path = path / 'history.jsonl'
    lines = (history_path.read_text(encoding='utf-8').splitlines()
             if history_path.exists() else [])
    while lines and not lines[-1].strip():
        lines.pop()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError('Corrupt history before the final line') from None
            # A torn final append is recoverable from the checkpoint record.
    epoch = int(checkpoint['epoch'])
    # An interrupted append may have written a complete row after the last
    # checkpoint. That epoch will be trained again, so discard the extra row.
    rows = [row for row in rows if row['epoch'] <= epoch]
    epochs = [row['epoch'] for row in rows]
    if epochs == list(range(1, epoch + 1)):
        pass
    elif epochs == list(range(1, epoch)) and checkpoint.get('record', {}).get('epoch') == epoch:
        rows.append(checkpoint['record'])
    else:
        raise ValueError('History/checkpoint epochs cannot be reconciled safely')
    history_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                    for row in rows), encoding='utf-8')


def fit(args):
    seed_all(args.seed)
    if args.width is None:
        args.width = {'marker_nafnet': 32, 'smp_resnet34_unet': 16}.get(args.architecture, 96)
    if args.target_marker is not None and args.architecture not in ('marker_nafnet', 'smp_resnet34_unet'):
        raise ValueError('--target-marker requires marker_nafnet or smp_resnet34_unet')
    active_markers = ((args.target_marker,) if args.target_marker else MARKERS)
    if (args.epochs < 1 or args.batch_size < 1
            or not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError('epochs, batch size and lr must be positive')
    if (not math.isfinite(args.cd68_weight) or args.cd68_weight <= 0
            or args.width < 4 or args.width % 4):
        raise ValueError('cd68 weight must be positive; width must be a multiple of 4')
    if args.architecture == 'prototype_marker':
        if args.num_shared_prototypes < 1 or args.num_task_prototypes < 1:
            raise ValueError('Prototype counts must be positive')
        if (not math.isfinite(args.prototype_temperature)
                or args.prototype_temperature <= 0):
            raise ValueError('Prototype temperature must be finite and positive')
        if (not math.isfinite(args.prototype_diversity_weight)
                or args.prototype_diversity_weight < 0):
            raise ValueError('Prototype diversity weight must be finite and nonnegative')
    if not (0 <= args.warmup_epochs < args.epochs) or args.eval_every < 1:
        raise ValueError('Invalid warmup/evaluation interval')

    manifest = load_manifest(args.manifest, args.data_root)
    all_names = sorted(sum((manifest['splits'][part] for part in ('train', 'val', 'holdout')), []))
    all_rois = sorted({roi_id(name) for name in all_names})
    folds = make_folds(all_names, args.folds, args.seed)
    recipe = recipe_from_args(args)
    selection = None
    if args.stage == 'dev':
        if args.fold is None or args.fold not in range(args.folds) or args.selection:
            raise ValueError('Dev stage requires one --fold and no --selection')
        val_rois = set(folds[args.fold])
        train_names = [name for name in all_names if roi_id(name) not in val_rois]
        val_names = [name for name in all_names if roi_id(name) in val_rois]
        stop_epochs = args.epochs
    else:
        if args.fold is not None or not args.selection:
            raise ValueError('Final stage requires --selection and no --fold')
        selection_path = Path(args.selection)
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        if (selection.get('kind') != 'semifinal_v7_cv_selection'
                or selection.get('split_sha256') != manifest['sha256']
                or selection.get('recipe') != recipe
                or selection.get('markers') != list(active_markers)
                or selection.get('all_rois') != all_rois
                or selection.get('source_sha256') != source_hashes()
                or selection.get('environment_source_sha256')
                   != environment()['source_sha256']):
            raise ValueError('Selection does not match this manifest and complete recipe')
        table = selection.get('epoch_table', [])
        if (not table or selection.get('selected_metrics') != max(
                table, key=lambda row: (row['ssim'], -row['epoch']))
                or selection.get('selected_epoch') != selection['selected_metrics']['epoch']):
            raise ValueError('Selection epoch does not follow its recorded decision rule')
        train_names, val_names = all_names, []
        stop_epochs = int(selection['selected_epoch'])
        if not 1 <= stop_epochs <= args.epochs:
            raise ValueError('Selected epoch is outside the development schedule')

    train_rois = sorted({roi_id(name) for name in train_names})
    actual_val_rois = sorted({roi_id(name) for name in val_names})
    if set(train_names) & set(val_names) or set(train_rois) & set(actual_val_rois):
        raise ValueError('Training and evaluation ROI overlap')
    if not train_names or (args.stage == 'dev' and not val_names):
        raise ValueError('Empty training/evaluation split')

    output = Path(args.output)
    if args.resume:
        if Path(args.resume).name != 'last.pt':
            raise ValueError('Resume only from last.pt, never from a best/eval snapshot')
        if Path(args.resume).resolve().parent != output.resolve():
            raise ValueError('Resume checkpoint must be in the same output directory')
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model_config = model_config_from_args(args, active_markers)
    model = build_reconstruction_model(model_config).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(
        'cuda', enabled=device.type == 'cuda' and not torch.cuda.is_bf16_supported()
    )
    weights = torch.tensor([args.cd68_weight if marker == 'CD68' else 1.0
                            for marker in active_markers])
    train_loader = make_training_loader(args, train_names, device, active_markers)
    val_loader = None
    if val_names:
        val_dataset = PairedMarkers(args.data_root, val_names, augment=False,
                                    cache=not args.no_cache)
        if active_markers != MARKERS:
            val_dataset = MarkerSubsetDataset(val_dataset, active_markers)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
            pin_memory=device.type == 'cuda',
        )

    run = {
        'kind': f'semifinal_v7_{args.stage}',
        'protocol': 'ROI out-of-fold development' if val_names else 'locked full-data refit',
        'markers': list(active_markers), 'model_config': model_config,
        'split_sha256': manifest['sha256'], 'all_rois': all_rois,
        'fold_assignment': folds, 'fold': args.fold,
        'train_names': train_names, 'val_names': val_names,
        'train_rois': train_rois, 'val_rois': actual_val_rois,
        'recipe': recipe, 'source_sha256': source_hashes(),
        'environment': environment(),
        'selection_sha256': (hashlib.sha256(Path(args.selection).read_bytes()).hexdigest()
                             if selection else None),
        'stopped_at_epoch': stop_epochs,
        'parameters': sum(p.numel() for p in model.parameters()),
    }

    start_epoch, steps, best_ssim = 0, 0, -math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        old_run = checkpoint.get('run', {})
        for key in ('kind', 'model_config', 'split_sha256', 'train_names', 'val_names',
                    'recipe', 'source_sha256', 'selection_sha256'):
            if old_run.get(key) != run[key]:
                raise ValueError(f'Resume mismatch: {key}')
        if old_run.get('environment', {}).get('source_sha256') != run['environment']['source_sha256']:
            raise ValueError('Resume mismatch: dependency/source environment')
        model.load_state_dict(checkpoint['model'], strict=True)
        ema.load_state_dict(checkpoint['ema'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']
        steps, best_ssim = checkpoint['steps'], checkpoint['best_ssim']
        torch.set_rng_state(checkpoint['rng']['torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        repair_history(output, checkpoint)
    else:
        write_json(output / 'run.json', run)
        write_json(output / 'split.json', manifest)

    def snapshot(epoch, record):
        return {
            'format': 1, 'run': run, 'epoch': epoch, 'steps': steps,
            'best_ssim': best_ssim,
            'record': record,
            'model': model.state_dict(), 'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
            'rng': {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            },
        }

    print(f'{args.stage}: train={len(train_names)} ({len(train_rois)} ROI), '
          f'val={len(val_names)} ({len(actual_val_rois)} ROI), epochs={stop_epochs}', flush=True)
    for epoch in range(start_epoch, stop_epochs):
        train_loader.generator.manual_seed(args.seed + epoch)
        random.seed(args.seed + epoch)
        np.random.seed(args.seed + epoch)
        train_loader.dataset.aug_strength = augmentation_strength(args, epoch)
        model.train()
        t0, total_loss, total_reconstruction, total_diversity, count, skipped = (
            time.perf_counter(), 0.0, 0.0, 0.0, 0, 0
        )
        for batch_index, (x, y, _) in enumerate(train_loader):
            lr = learning_rate(args, epoch, batch_index / len(train_loader))
            for group in optimizer.param_groups:
                group['lr'] = lr
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                pred = model(x)
            data_loss = reconstruction_loss(pred, y, weights, args.loss)
            diversity = (model.prototype_diversity_loss()
                         if args.architecture == 'prototype_marker' else None)
            loss = (data_loss + args.prototype_diversity_weight * diversity
                    if diversity is not None else data_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch + 1}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                            error_if_nonfinite=not scaler.is_enabled())
            if not torch.isfinite(norm):
                # GradScaler skips this FP16 update and reduces its scale.
                scaler.step(optimizer)
                scaler.update()
                skipped += 1
                if skipped > 10:
                    raise FloatingPointError('Repeated FP16 overflow; use BF16 or a smaller batch')
                continue
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            decay = min(0.995, (1 + steps) / (10 + steps))
            update_ema(ema, model, decay)
            total_loss += float(loss.item()) * len(x)
            total_reconstruction += float(data_loss.item()) * len(x)
            if diversity is not None:
                total_diversity += float(diversity.item()) * len(x)
            count += len(x)

        if count == 0:
            raise FloatingPointError(f'No successful optimizer updates in epoch {epoch + 1}')
        record = {
            'epoch': epoch + 1, 'loss': total_loss / count, 'lr': lr,
            'aug_strength': train_loader.dataset.aug_strength,
            'skipped_overflow_batches': skipped,
            'seconds': time.perf_counter() - t0,
        }
        if args.architecture == 'prototype_marker':
            record['reconstruction_loss'] = total_reconstruction / count
            record['prototype_diversity'] = total_diversity / count
        improved = False
        if val_loader is not None and ((epoch + 1) % args.eval_every == 0
                                        or epoch + 1 == stop_epochs):
            metrics = evaluate(ema, val_loader, device, tta=args.tta,
                               marker_names=active_markers)
            record['validation'] = compact(metrics)
            improved = metrics['ssim'] > best_ssim
            if improved:
                best_ssim = metrics['ssim']
                write_json(output / 'best_validation.json', metrics)
        state = snapshot(epoch + 1, record)
        if 'validation' in record:
            # EMA-only checkpoints retain the exact pooled-CV selected epoch
            # without duplicating optimizer state at every evaluation point.
            eval_path = output / f'eval_epoch_{epoch + 1:03d}.pt'
            torch.save({'format': 1, 'run': run, 'epoch': epoch + 1,
                        'ema': ema.state_dict()}, eval_path.with_suffix('.tmp'))
            eval_path.with_suffix('.tmp').replace(eval_path)
            write_json(output / f'eval_epoch_{epoch + 1:03d}.json', metrics)
        if improved:
            torch.save(state, output / 'best.tmp')
            (output / 'best.tmp').replace(output / 'best.pt')
        # Commit last.pt only after all artifacts of this epoch are durable.
        # On resume it is the authoritative completed-epoch marker.
        temporary = output / 'last.tmp'
        torch.save(state, temporary)
        temporary.replace(output / 'last.pt')
        with (output / 'history.jsonl').open('a', encoding='utf-8') as history:
            history.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"epoch={epoch + 1} loss={record['loss']:.5f} "
              f"val_ssim={record.get('validation', {}).get('ssim', float('nan')):.5f}",
              flush=True)
    if args.stage == 'final':
        # Same checkpoint format as src.train_marker_context.load_model/infer.
        shutil.copy2(output / 'last.pt', output / 'final.pt')
    print(f'Completed: {output}', flush=True)


def summarize(args):
    runs = []
    for path in args.runs:
        directory = Path(path)
        run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
        history = [json.loads(line) for line in
                   (directory / 'history.jsonl').read_text(encoding='utf-8').splitlines()
                   if line.strip()]
        if run.get('kind') != 'semifinal_v7_dev':
            raise ValueError(f'Not a development run: {directory}')
        runs.append((directory, run, history))
    if not runs:
        raise ValueError('Supply all development fold directories')
    first = runs[0][1]
    if first['source_sha256'] != source_hashes():
        raise ValueError('Current source differs from the development runs')
    expected_folds = set(range(first['recipe']['folds']))
    if len(runs) != len(expected_folds) or {r['fold'] for _, r, _ in runs} != expected_folds:
        raise ValueError('Exactly one completed run per fold is required')
    seen_rois = set()
    common_epochs = None
    for directory, run, history in runs:
        if (run['recipe'] != first['recipe'] or run['markers'] != first['markers']
                or run['split_sha256'] != first['split_sha256']
                or run['all_rois'] != first['all_rois']
                or run['fold_assignment'] != first['fold_assignment']
                or run['source_sha256'] != first['source_sha256']
                or run['environment']['source_sha256']
                   != first['environment']['source_sha256']):
            raise ValueError(f'Unmatched recipe, source or manifest: {directory}')
        if (run['val_rois'] != run['fold_assignment'][run['fold']]
                or set(run['train_rois']) | set(run['val_rois']) != set(run['all_rois'])
                or {roi_id(name) for name in run['train_names']} != set(run['train_rois'])
                or {roi_id(name) for name in run['val_names']} != set(run['val_rois'])):
            raise ValueError(f'Fold assignment does not match sample lists: {directory}')
        if set(run['train_rois']) & set(run['val_rois']):
            raise ValueError(f'Training/validation ROI overlap: {directory}')
        if seen_rois & set(run['val_rois']):
            raise ValueError(f'Validation ROI repeated across folds: {directory}')
        seen_rois.update(run['val_rois'])
        if ([row['epoch'] for row in history]
                != list(range(1, run['recipe']['schedule_epochs'] + 1))):
            raise ValueError(f'Incomplete training history: {directory}')
        epochs = {row['epoch'] for row in history if 'validation' in row}
        common_epochs = epochs if common_epochs is None else common_epochs & epochs
    if seen_rois != set(first['all_rois']) or not common_epochs:
        raise ValueError('Validation folds do not cover every ROI and evaluation epoch')

    epoch_table = []
    for epoch in sorted(common_epochs):
        fold_values = []
        for _, run, history in runs:
            row = next(row for row in history if row['epoch'] == epoch)
            val = row['validation']
            if val['count'] != len(run['val_names']):
                raise ValueError('Validation count mismatch')
            if set(val['rois']) != set(run['val_rois']):
                raise ValueError('Validation ROI mismatch')
            fold_values.append(val)
        count = sum(value['count'] for value in fold_values)
        marker_metrics = {
            marker: {
                metric: sum(value['markers'][marker][metric] * value['count']
                            for value in fold_values) / count
                for metric in ('ssim', 'psnr')
            }
            for marker in first['markers']
        }
        roi_metrics = {}
        for value in fold_values:
            roi_metrics.update(value['rois'])
        epoch_table.append({
            'epoch': epoch, 'count': count,
            'ssim': sum(value['ssim'] * value['count'] for value in fold_values) / count,
            'psnr': sum(value['psnr'] * value['count'] for value in fold_values) / count,
            'markers': marker_metrics, 'rois': roi_metrics,
        })
    selected = max(epoch_table, key=lambda row: (row['ssim'], -row['epoch']))
    for directory, _, _ in runs:
        if not (directory / f"eval_epoch_{selected['epoch']:03d}.pt").exists():
            raise ValueError(f'Missing selected-epoch EMA checkpoint: {directory}')
    selection = {
        'kind': 'semifinal_v7_cv_selection',
        'markers': first['markers'],
        'split_sha256': first['split_sha256'], 'all_rois': first['all_rois'],
        'source_sha256': first['source_sha256'],
        'environment_source_sha256': first['environment']['source_sha256'],
        'recipe': first['recipe'], 'selected_epoch': selected['epoch'],
        'selection_rule': 'maximum pooled out-of-fold mean SSIM; PSNR reported separately',
        'selected_metrics': selected, 'epoch_table': epoch_table,
        'fold_directories': [str(directory.resolve()) for directory, _, _ in runs],
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, selection)
    print(f"selected_epoch={selected['epoch']} OOF_ssim={selected['ssim']:.5f} "
          f"OOF_psnr={selected['psnr']:.3f} output={output}", flush=True)


def infer(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    run = checkpoint.get('run', {})
    if (run.get('kind') != 'semifinal_v7_final'
            or run.get('source_sha256') != source_hashes()
            or checkpoint.get('epoch') != run.get('stopped_at_epoch')):
        raise ValueError('Expected a complete semifinal V7 final checkpoint from this source')
    args.tta = run['recipe']['tta']
    args.seed = run['recipe']['seed']
    infer_command(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    train = subparsers.add_parser('fit', help='Train one ROI fold or the locked final refit')
    train.add_argument('--stage', choices=('dev', 'final'), default='dev')
    train.add_argument('--data-root', required=True)
    train.add_argument('--manifest', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--selection', help='Selection JSON from summarize; final stage only')
    train.add_argument('--fold', type=int, help='Development fold index, zero based')
    train.add_argument('--folds', type=int, default=5)
    train.add_argument('--augmentation', choices=('legacy-v6', 'geometry', 'dapi-noise'),
                       default='geometry')
    train.add_argument('--loss', choices=('legacy-v6', 'normalized'), default='normalized')
    train.add_argument('--architecture', choices=('context', 'marker_specific',
                                                  'prototype_marker', 'marker_nafnet',
                                                  'smp_resnet34_unet') + _NEW_CHANNEL_ARCHITECTURES,
                        default='context')
    train.add_argument('--target-marker', choices=MARKERS,
                       help='Train one marker (marker_nafnet or smp_resnet34_unet); default is all four')
    train.add_argument('--width', type=int,
                       help='Default: 16 for smp_resnet34_unet, 32 for marker_nafnet, 96 otherwise')
    train.add_argument('--num-shared-prototypes', type=int, default=8)
    train.add_argument('--num-task-prototypes', type=int, default=4)
    train.add_argument('--prototype-temperature', type=float, default=0.25)
    train.add_argument('--prototype-diversity-weight', type=float, default=0.001)
    train.add_argument('--epochs', type=int, default=300,
                       help='Full learning-rate schedule; final refit stops at selected epoch')
    train.add_argument('--eval-every', type=int, default=50)
    train.add_argument('--batch-size', type=int, default=8)
    train.add_argument('--lr', type=float, default=3e-4)
    train.add_argument('--cd68-weight', type=float, default=3.0)
    train.add_argument('--warmup-epochs', type=int, default=10)
    train.add_argument('--weaken-start-epoch', type=int, default=200)
    train.add_argument('--tta', type=int, choices=(1, 4, 8), default=4)
    train.add_argument('--seed', type=int, choices=(SEMIFINAL_SEED,), default=SEMIFINAL_SEED)
    train.add_argument('--device', default='cuda')
    train.add_argument('--no-cache', action='store_true')
    train.add_argument('--resume', help='Resume last.pt from the same output directory')
    summary = subparsers.add_parser('summarize', help='Select one epoch from all ROI folds')
    summary.add_argument('--runs', nargs='+', required=True)
    summary.add_argument('--output', required=True)
    inference = subparsers.add_parser('infer', help='Infer with the final run\'s locked TTA')
    inference.add_argument('--checkpoint', required=True)
    inference.add_argument('--input', required=True, help='Official test/DAPI directory')
    inference.add_argument('--output', required=True)
    inference.add_argument('--device', default='cuda')
    inference.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'fit':
        fit(args)
    elif args.command == 'summarize':
        summarize(args)
    else:
        infer(args)


if __name__ == '__main__':
    main()
