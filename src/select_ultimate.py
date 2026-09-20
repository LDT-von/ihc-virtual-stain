"""Select residual branches only on development validation, then lock inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .data.roi_manifest import MARKERS, digest, selected_names
from .models.anchored_ihc import AnchoredIHC
from .train_marker_context import compact, environment, evaluate, load_manifest, load_model, make_loader, seed_all, write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_markers(baseline, candidate):
    if baseline['count'] != candidate['count']:
        raise ValueError('Baseline and candidate counts differ')
    mask = []
    for marker in MARKERS:
        base, refined = baseline['markers'][marker], candidate['markers'][marker]
        if not all(math.isfinite(row[key]) for row in (base, refined) for key in ('ssim', 'psnr')):
            raise ValueError('Non-finite validation metric')
        # Default to baseline on ties. No invented PSNR normalization/composite.
        mask.append(refined['ssim'] > base['ssim'] and refined['psnr'] >= base['psnr'])
    return mask


def apply_deployment(model, checkpoint, checkpoint_path, path, tta, jpeg_quality, allow_smoke=False):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    expected = digest({key: value for key, value in report.items() if key != 'sha256'})
    if report.get('format') != 1 or report.get('sha256') != expected:
        raise ValueError('Invalid deployment manifest/checksum')
    if (report.get('inference_source_sha256') != environment()['source_sha256']
            or report.get('selector_source_sha256') != sha256(__file__)):
        raise ValueError('Deployment source changed; preserve the validated code bundle')
    if not isinstance(model, AnchoredIHC) or report['checkpoint_sha256'] != sha256(checkpoint_path):
        raise ValueError('Deployment checkpoint mismatch')
    if report['split_sha256'] != checkpoint['run']['split_sha256'] or tuple(report['markers']) != MARKERS:
        raise ValueError('Deployment split/marker mismatch')
    if (tta, jpeg_quality) != (report['tta'], report['jpeg_quality']):
        raise ValueError('Inference must preserve validation TTA and JPEG quality')
    if report['smoke_only'] and not allow_smoke:
        raise ValueError('Smoke deployment is not a competition submission')
    mask = report['marker_mask']
    if len(mask) != len(MARKERS) or any(type(value) is not bool for value in mask):
        raise ValueError('Invalid deployment marker mask')
    model.deployment_mask = mask
    return report


def select(args):
    seed_all(args.seed)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Use a new selection output; do not overwrite a locked decision')
    manifest = load_manifest(args.manifest, args.data_root)
    model, checkpoint = load_model(args.checkpoint, torch.device(args.device))
    if not isinstance(model, AnchoredIHC):
        raise ValueError('Selection requires an anchored residual checkpoint')
    if checkpoint['run']['split_sha256'] != manifest['sha256']:
        raise ValueError('Checkpoint/split mismatch')
    run = checkpoint['run']
    inference_source = environment()['source_sha256']
    if run['environment']['source_sha256'] != inference_source:
        raise ValueError('Candidate source changed since training; use its recorded code')
    smoke = bool(args.limit or run['args'].get('train_limit') or run['args'].get('val_limit')
                 or run['baseline_provenance']['run']['args'].get('train_limit')
                 or run['baseline_provenance']['run']['args'].get('val_limit'))
    if smoke and not args.allow_smoke:
        raise ValueError('Subset training/selection is smoke-only; use --allow-smoke only for pipeline tests')
    names = selected_names(manifest['splits']['val'], args.limit, args.seed)
    if set(names) & set(run['train_names']):
        raise ValueError('Validation samples were used for training')
    loader = make_loader(args.data_root, names, args.batch_size, cache=not args.no_cache)
    device = torch.device(args.device)
    baseline = evaluate(model.baseline, loader, device, args.tta, args.jpeg_quality)
    candidate = evaluate(model, loader, device, args.tta, args.jpeg_quality)
    if [r['name'] for r in baseline['rows']] != [r['name'] for r in candidate['rows']]:
        raise ValueError('Paired evaluation order mismatch')
    mask = choose_markers(baseline, candidate)
    model.deployment_mask = mask
    deployed = evaluate(model, loader, device, args.tta, args.jpeg_quality)
    for i, marker in enumerate(MARKERS):
        expected = candidate if mask[i] else baseline
        for key in ('ssim', 'psnr'):
            if not np.isclose(deployed['markers'][marker][key], expected['markers'][marker][key], atol=2e-6, rtol=0):
                raise AssertionError('Selected model differs from independently evaluated branch')
    report = {'format': 1, 'markers': list(MARKERS), 'checkpoint_sha256': sha256(args.checkpoint),
              'split_sha256': manifest['sha256'], 'selection_split': 'val',
              'validation_names_sha256': digest(names), 'tta': args.tta, 'jpeg_quality': args.jpeg_quality,
              'marker_mask': mask, 'smoke_only': smoke,
              'rule': 'enable marker refinement iff validation SSIM increases and PSNR does not decrease',
              'baseline': compact(baseline), 'candidate': compact(candidate), 'deployed': compact(deployed),
              'rows': {'baseline': baseline['rows'], 'candidate': candidate['rows'], 'deployed': deployed['rows']},
              'interpretation': 'Development model selection only; no test-score or unseen-data guarantee',
              'inference_source_sha256': inference_source,
              'selector_source_sha256': sha256(__file__)}
    report['sha256'] = digest(report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, report)
    print(json.dumps({'marker_mask': mask, 'baseline': compact(baseline), 'deployed': compact(deployed),
                      'smoke_only': smoke}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tta', type=int, choices=(1,4,8), default=4)
    p.add_argument('--jpeg-quality', type=int, choices=range(1,101), default=95)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--no-cache', action='store_true')
    p.add_argument('--allow-smoke', action='store_true')
    select(p.parse_args())


if __name__ == '__main__':
    main()
