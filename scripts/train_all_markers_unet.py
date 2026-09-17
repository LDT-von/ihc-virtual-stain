#!/usr/bin/env python3
"""Train all 4 markers sequentially with per-marker U-Net models.

Usage:
    python scripts/train_all_markers_unet.py \\
        --data-root /data1/AIC \\
        --manifest /data1/AIC/configs/roi_split_v1.json \\
        --run-dir checkpoints/per_marker_unet \\
        --model-type standard \\
        --epochs 60 \\
        --batch-size 16 \\
        --device cuda

Then generate test predictions:
    python scripts/train_all_markers_unet.py --mode predict \\
        --data-root /data1/AIC \\
        --run-dir checkpoints/per_marker_unet \\
        --output results/per_marker_unet
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']


def run_command(cmd, env=None):
    """Run command and stream output."""
    print(f'\n{"="*60}')
    print(f'Running: {" ".join(str(c) for c in cmd)}')
    print('='*60, flush=True)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f'Command failed with code {result.returncode}')
        return False
    return True


def train_all(args):
    """Train all 4 markers."""
    results = {}
    total_start = time.time()
    
    for i, marker in enumerate(MARKERS):
        marker_dir = Path(args.run_dir) / marker
        marker_dir.mkdir(parents=True, exist_ok=True)
        
        print(f'\n{"#"*60}')
        print(f'#{i+1}/4: Training {marker}')
        print(f'# Output: {marker_dir}')
        print(f'#' * 60, flush=True)
        
        cmd = [
            sys.executable, '-m', 'src.train_marker_unet', 'train',
            '--marker', marker,
            '--data-root', args.data_root,
            '--manifest', args.manifest,
            '--output', str(marker_dir),
            '--model-type', args.model_type,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--lr', str(args.lr),
            '--device', args.device,
            '--seed', str(args.seed),
            '--tta', str(args.tta),
        ]
        
        t0 = time.time()
        success = run_command(cmd)
        elapsed = time.time() - t0
        
        # Read validation result
        val_file = marker_dir / 'best_validation.json'
        if val_file.exists():
            val = json.loads(val_file.read_text())
            results[marker] = {
                'ssim': val['ssim'],
                'psnr': val['psnr'],
                'count': val['count'],
                'elapsed': elapsed,
                'success': success,
            }
        else:
            results[marker] = {'success': success, 'elapsed': elapsed}
    
    # Summary
    total_elapsed = time.time() - total_start
    print(f'\n{"="*60}')
    print('TRAINING SUMMARY')
    print('='*60)
    print(f'{"Marker":<12} {"SSIM":>8} {"PSNR":>8} {"Time":>8}')
    print('-'*60)
    
    total_ssim = 0
    total_psnr = 0
    for marker in MARKERS:
        r = results.get(marker, {})
        ssim = r.get('ssim', 0)
        psnr = r.get('psnr', 0)
        elapsed = r.get('elapsed', 0)
        total_ssim += ssim
        total_psnr += psnr
        print(f'{marker:<12} {ssim:>8.4f} {psnr:>8.2f} {elapsed:>7.0f}s')
    
    print('-'*60)
    avg_ssim = total_ssim / len(MARKERS)
    avg_psnr = total_psnr / len(MARKERS)
    print(f'{"AVG":<12} {avg_ssim:>8.4f} {avg_psnr:>8.2f} {total_elapsed:>7.0f}s')
    print(f'{"="*60}')
    print(f'Total time: {total_elapsed/60:.1f} minutes')
    
    # Estimate platform score
    # Platform: 0.7*SSIM + 0.3*Norm(PSNR)
    # Normalized PSNR: typically ~0.7-0.9 for range 20-26
    estimated = 0.7 * avg_ssim * 100 + 0.3 * (avg_psnr / 26 * 100)
    print(f'Estimated platform score: ~{estimated:.1f}')
    
    # Save summary
    summary = {
        'results': results,
        'avg_ssim': avg_ssim,
        'avg_psnr': avg_psnr,
        'estimated_score': estimated,
        'total_elapsed': total_elapsed,
    }
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    Path(args.run_dir, 'summary.json').write_text(json.dumps(summary, indent=2))
    
    return results


def predict_all(args):
    """Generate predictions for all 4 markers."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    
    for marker in MARKERS:
        ckpt = Path(args.run_dir) / marker / 'best.pt'
        if not ckpt.exists():
            print(f'Warning: {ckpt} not found, skipping {marker}')
            continue
        
        print(f'\nInferring {marker} -> {output}', flush=True)
        
        cmd = [
            sys.executable, '-m', 'src.train_marker_unet', 'infer',
            '--checkpoint', str(ckpt),
            '--input', str(Path(args.data_root) / 'test'),
            '--output', str(output),
            '--batch-size', str(args.batch_size),
            '--device', args.device,
            '--tta', str(args.tta),
            '--jpeg-quality', str(args.jpeg_quality),
        ]
        run_command(cmd)
    
    # Pack into submission zip
    zip_name = output / 'submission.zip'
    print(f'\nPacking submissions into {zip_name}', flush=True)
    
    import zipfile
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for marker in MARKERS:
            marker_dir = output / 'results' / 'test' / marker
            if marker_dir.exists():
                for f in sorted(marker_dir.glob('*.jpg')):
                    zf.write(f, f.relative_to(output))
    
    print(f'Packed: {zip_name} ({zip_name.stat().st_size / 1024 / 1024:.1f} MB)')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    
    p_train = sub.add_parser('train')
    p_train.add_argument('--data-root', required=True)
    p_train.add_argument('--manifest', required=True)
    p_train.add_argument('--run-dir', required=True)
    p_train.add_argument('--model-type', default='standard',
                        choices=['light', 'standard', 'deep', 'deep_large'])
    p_train.add_argument('--epochs', type=int, default=60)
    p_train.add_argument('--batch-size', type=int, default=16)
    p_train.add_argument('--lr', type=float, default=5e-4)
    p_train.add_argument('--device', default='cuda')
    p_train.add_argument('--seed', type=int, default=42)
    p_train.add_argument('--tta', type=int, default=4)
    
    p_predict = sub.add_parser('predict')
    p_predict.add_argument('--data-root', required=True)
    p_predict.add_argument('--run-dir', required=True)
    p_predict.add_argument('--output', required=True)
    p_predict.add_argument('--batch-size', type=int, default=32)
    p_predict.add_argument('--device', default='cuda')
    p_predict.add_argument('--seed', type=int, default=42)
    p_predict.add_argument('--tta', type=int, default=4)
    p_predict.add_argument('--jpeg-quality', type=int, default=100)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_all(args)
    elif args.mode == 'predict':
        predict_all(args)


if __name__ == '__main__':
    main()
