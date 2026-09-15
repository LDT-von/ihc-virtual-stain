"""Portable server launcher. Nothing is uploaded or submitted automatically."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True, help='Directory directly containing train/DAPI and test/DAPI')
    p.add_argument('--mode', choices=('smoke', 'train', 'ablation', 'refit', 'predict'), default='train')
    p.add_argument('--run-dir', type=Path, default=ROOT/'checkpoints'/'mcn_server_v1')
    p.add_argument('--manifest', type=Path, default=ROOT/'configs'/'roi_split_v1.json')
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--tta', type=int, choices=(1, 4, 8), default=4)
    p.add_argument('--no-cache', action='store_true', help='For machines with limited host RAM')
    args = p.parse_args()
    args.data_root, args.run_dir, args.manifest = args.data_root.resolve(), args.run_dir.resolve(), args.manifest.resolve()
    if not (args.data_root/'train'/'DAPI').is_dir():
        p.error('--data-root must directly contain train/DAPI')
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    if args.resume:
        args.resume = args.resume.resolve()

    def run(*parts):
        command = [sys.executable, '-m', 'src.train_marker_context', *map(str, parts)]
        print('Running:', subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    if args.mode == 'predict':
        if args.checkpoint is None:
            p.error('predict requires --checkpoint')
        run('infer', '--input', args.data_root/'test'/'DAPI', '--checkpoint', args.checkpoint,
            '--output', args.run_dir, '--tta', args.tta, '--batch-size', args.batch_size)
        return
    if not args.manifest.exists():
        run('split', '--data-root', args.data_root, '--output', args.manifest, '--seed', args.seed)
    options = ['--data-root', args.data_root, '--manifest', args.manifest,
               '--output', args.run_dir, '--batch-size', args.batch_size, '--seed', args.seed,
               '--width', args.width, '--lr', args.lr, '--epochs', args.epochs]
    if args.no_cache:
        options += ['--no-cache']
    if args.resume:
        options += ['--resume', args.resume]
    if args.mode == 'smoke':
        options = ['--data-root', args.data_root, '--manifest', args.manifest, '--output', args.run_dir,
                   '--width', '8', '--epochs', '2', '--batch-size', '2', '--train-limit', '32', '--val-limit', '8']
    elif args.mode == 'ablation':
        options += ['--no-context']
    elif args.mode == 'refit':
        if args.checkpoint is None and args.resume is None:
            p.error('refit requires --checkpoint (validated best.pt) or --resume')
        options += ['--full-data']
        if args.checkpoint:
            options += ['--init-checkpoint', args.checkpoint]
    run('train', *options)
    if args.mode == 'smoke':
        run('eval', '--data-root', args.data_root, '--manifest', args.manifest,
            '--checkpoint', args.run_dir/'best.pt', '--split', 'val', '--limit', '8', '--tta', '8',
            '--output', args.run_dir/'smoke_jpeg_tta8.json')
    elif args.mode in ('train', 'ablation'):
        results = []
        for tta in (1, 4, 8):
            destination = args.run_dir/f'val_jpeg_tta{tta}.json'
            run('eval', '--data-root', args.data_root, '--manifest', args.manifest,
                '--checkpoint', args.run_dir/'best.pt', '--split', 'val', '--tta', tta,
                '--batch-size', args.batch_size, '--output', destination)
            scores = json.loads(destination.read_text(encoding='utf-8'))
            results.append({k: scores[k] for k in ('tta', 'ssim', 'psnr', 'count')})
        report = {'criterion': 'maximum validation SSIM; NOT official composite score',
                  'candidates': results, 'selected': max(results, key=lambda x: x['ssim']),
                  'note': 'Lock architecture and TTA before evaluating holdout once. No automatic full-data refit.'}
        (args.run_dir/'tta_selection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
