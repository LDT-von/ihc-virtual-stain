"""MCN baseline -> frozen cross-marker refinement -> validation gate -> submission."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=ROOT/'configs/roi_split_v1.json')
    p.add_argument('--mode', choices=('train', 'smoke', 'evaluate', 'predict'), default='train')
    p.add_argument('--baseline-checkpoint', type=Path, help='Verified multi-marker baseline on this exact manifest')
    p.add_argument('--baseline-epochs', type=int, default=60)
    p.add_argument('--refine-epochs', type=int, default=20)
    p.add_argument('--baseline-width', type=int, default=32)
    p.add_argument('--width', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--tta', choices=(1,4,8), type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--resume', action='store_true', help='Resume the unchanged staged recipe')
    p.add_argument('--output', type=Path, help='Required for predict; new output directory')
    args = p.parse_args()
    args.data_root, args.run_dir, args.manifest = [path.resolve() for path in (args.data_root,args.run_dir,args.manifest)]

    def run(module, *parts):
        command = [sys.executable, '-m', module, *map(str, parts)]
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    if args.mode == 'evaluate':
        options = ['--no-cache'] if args.no_cache else []
        run('src.evaluate_ultimate', '--data-root', args.data_root, '--manifest', args.manifest,
            '--checkpoint', args.run_dir/'refiner/best.pt', '--deployment', args.run_dir/'deployment.json',
            '--output', args.run_dir/'holdout_locked.json', '--batch-size', args.batch_size,
            '--device', args.device, '--seed', args.seed, *options)
        return

    if args.mode == 'predict':
        if args.output is None:
            p.error('predict requires --output')
        decision = json.loads((args.run_dir/'deployment.json').read_text(encoding='utf-8'))
        if decision['smoke_only']:
            p.error('Smoke model cannot generate a competition submission')
        from src.data.roi_manifest import digest
        from src.select_ultimate import sha256
        holdout_path = args.run_dir/'holdout_locked.json'
        if not holdout_path.is_file():
            p.error('Run --mode evaluate once before packaging the locked model')
        holdout = json.loads(holdout_path.read_text(encoding='utf-8'))
        if (holdout.get('format') != 1 or holdout.get('split') != 'holdout' or holdout.get('smoke_only')
                or holdout.get('deployment_sha256') != decision['sha256']
                or holdout.get('checkpoint_sha256') != sha256(args.run_dir/'refiner/best.pt')
                or holdout.get('sha256') != digest({k:v for k,v in holdout.items() if k != 'sha256'})):
            p.error('Holdout report does not match this locked model')
        output = args.output.resolve()
        run('src.train_marker_context', 'infer', '--input', args.data_root/'test/DAPI',
            '--checkpoint', args.run_dir/'refiner/best.pt', '--deployment', args.run_dir/'deployment.json',
            '--tta', decision['tta'], '--jpeg-quality', decision['jpeg_quality'], '--output', output,
            '--batch-size', args.batch_size, '--device', args.device)
        from scripts.package_final import package
        print(json.dumps(package(args.data_root/'test/DAPI', output, output/'submission.zip')), flush=True)
        return

    from src.train_marker_context import load_manifest
    manifest = load_manifest(args.manifest, args.data_root)
    if args.baseline_checkpoint:
        args.baseline_checkpoint = args.baseline_checkpoint.resolve()
    if args.mode == 'smoke':
        args.baseline_epochs = args.refine_epochs = 2
        args.baseline_width = args.width = 8
    recipe = {key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()
              if key not in ('resume', 'output')}
    recipe['split_sha256'] = manifest['sha256']
    plan = args.run_dir/'recipe.json'
    if plan.exists():
        if not args.resume or json.loads(plan.read_text(encoding='utf-8')) != recipe:
            raise ValueError('Run exists; resume requires the identical recipe and --resume')
    elif args.resume:
        raise ValueError('No existing recipe to resume')
    elif args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError('Use an empty run directory')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps(recipe, indent=2), encoding='utf-8')
    common = ['--data-root', args.data_root, '--manifest', args.manifest, '--batch-size', args.batch_size,
              '--device', args.device, '--seed', args.seed, '--val-jpeg-quality', 95]
    if args.no_cache:
        common.append('--no-cache')
    if args.mode == 'smoke':
        common += ['--train-limit',32,'--val-limit',8]

    def stage(name, epochs, architecture, width, lr, baseline=None):
        folder = args.run_dir/name
        last = folder/'last.pt'
        options = ['--output',folder,'--epochs',epochs,'--architecture',architecture,'--width',width,'--lr',lr]
        if last.exists():
            if not args.resume:
                raise FileExistsError(last)
            # The trainer checks source, split and hyperparameters even for a completed stage.
            options += ['--resume',last]
        elif baseline:
            options += ['--baseline-checkpoint',baseline]
        run('src.train_marker_context','train',*common,*options)
        if not (folder/'best.pt').is_file():
            raise RuntimeError(f'No validated best checkpoint in {folder}')
        return folder/'best.pt'

    baseline = args.baseline_checkpoint or stage('baseline',args.baseline_epochs,'context',args.baseline_width,5e-4)
    if args.baseline_checkpoint:
        from src.select_ultimate import sha256
        marker = args.run_dir/'baseline_identity.json'
        identity = {'path':str(baseline), 'sha256':sha256(baseline)}
        if marker.exists() and json.loads(marker.read_text()) != identity:
            raise ValueError('External baseline changed since this run began')
        marker.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    candidate = stage('refiner',args.refine_epochs,'anchored',args.width,1e-4,baseline)
    selection = args.run_dir/'deployment.json'
    if not selection.exists():
        options = ['--limit',8,'--allow-smoke'] if args.mode == 'smoke' else []
        if args.no_cache:
            options.append('--no-cache')
        run('src.select_ultimate','--data-root',args.data_root,'--manifest',args.manifest,
            '--checkpoint',candidate,'--output',selection,'--tta',args.tta,'--device',args.device,
            '--batch-size',args.batch_size,*options)
    else:
        from src.select_ultimate import apply_deployment
        from src.train_marker_context import load_model
        model, ck = load_model(candidate,torch.device('cpu'))
        apply_deployment(model,ck,candidate,selection,args.tta,95,args.mode == 'smoke')
    print(f'Completed development recipe. Deployment decision: {selection}. Official score unmeasured.',flush=True)


if __name__ == '__main__':
    main()
