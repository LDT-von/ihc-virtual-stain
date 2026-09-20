"""Compare a locked model with its frozen baseline on untouched holdout ROIs."""
import argparse
import json
from pathlib import Path

import torch

from .data.roi_manifest import MARKERS, digest
from .select_ultimate import apply_deployment, sha256
from .train_marker_context import compact, evaluate, load_manifest, load_model, make_loader, seed_all, write_json


def evaluate_locked(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Holdout report exists; do not repeatedly tune against the holdout')
    seed_all(args.seed)
    manifest = load_manifest(args.manifest, args.data_root)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    decision = json.loads(Path(args.deployment).read_text(encoding='utf-8'))
    apply_deployment(model, checkpoint, args.checkpoint, args.deployment,
                     decision['tta'], decision['jpeg_quality'], args.allow_smoke)
    if checkpoint['run']['split_sha256'] != manifest['sha256']:
        raise ValueError('Checkpoint/split mismatch')
    names = manifest['splits']['holdout']
    if set(names) & set(checkpoint['run']['train_names']):
        raise ValueError('Holdout samples were used for training')
    loader = make_loader(args.data_root, names, args.batch_size, cache=not args.no_cache)
    baseline = evaluate(model.baseline, loader, device, decision['tta'], decision['jpeg_quality'])
    deployed = evaluate(model, loader, device, decision['tta'], decision['jpeg_quality'])
    if [row['name'] for row in baseline['rows']] != [row['name'] for row in deployed['rows']]:
        raise ValueError('Paired holdout evaluation order mismatch')
    delta = {marker: {metric: deployed['markers'][marker][metric]-baseline['markers'][marker][metric]
                      for metric in ('ssim', 'psnr')} for marker in MARKERS}
    report = {'format': 1, 'split': 'holdout', 'split_sha256': manifest['sha256'],
              'checkpoint_sha256': sha256(args.checkpoint), 'deployment_sha256': decision['sha256'],
              'holdout_names_sha256': digest(names), 'smoke_only': decision['smoke_only'],
              'baseline': compact(baseline), 'deployed': compact(deployed), 'delta': delta,
              'rows': {'baseline': baseline['rows'], 'deployed': deployed['rows']},
              'interpretation': 'Single locked ROI holdout comparison; no branch selection on holdout; not an official score'}
    report['sha256'] = digest(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps({'holdout_count': len(names), 'delta': delta, 'smoke_only': decision['smoke_only']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'manifest', 'checkpoint', 'deployment', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--allow-smoke', action='store_true')
    evaluate_locked(parser.parse_args())


if __name__ == '__main__':
    main()
