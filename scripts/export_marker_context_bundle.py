"""Export an explicit, data-free source bundle and check it in a fresh directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'src/__init__.py', 'src/train_marker_context.py', 'src/submit.py',
    'src/models/__init__.py', 'src/models/marker_context.py',
    'src/models/pix2pix_gan.py', 'src/models/losses.py',
    'src/models/flow_matching.py', 'src/models/unet.py',
    'src/data/__init__.py', 'src/data/roi_manifest.py', 'src/data/dataset.py',
    'src/metrics/ssim_psnr.py', 'inference_8x_tta.py', 'tta_eval.py',
    'configs/roi_split_v1.json', 'requirements-marker-context.txt', 'requirements.txt',
    'scripts/run_marker_context.py', 'scripts/audit_saved_predictions.py',
    'scripts/benchmark_marker_baseline.py', 'scripts/export_marker_context_bundle.py',
    'tests/test_marker_context.py', 'docs/MARKER_CONTEXT_V1.md',
    'docs/experiments/saved_prediction_audit_20260915.json',
    'docs/experiments/roi_ridge_baseline.json',
]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'artifacts'/'marker_context_v1_20260915.zip')
    args = p.parse_args()
    payload = {name: (ROOT/name).read_bytes() for name in FILES}
    manifest = {'kind': 'source handoff; no data, checkpoints, or official predictions',
                'instructions': 'Read docs/MARKER_CONTEXT_V1.md. Merge carefully if applying to an existing checkout.',
                'files': {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, data in payload.items():
            z.writestr(name, data)
        z.writestr('BUNDLE_MANIFEST.json', json.dumps(manifest, indent=2))
    with tempfile.TemporaryDirectory(prefix='mcn_bundle_check_') as temp:
        with zipfile.ZipFile(args.output) as z:
            assert all(not n.startswith(('/', '\\')) and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(temp)
        env = dict(os.environ, PYTHONPATH='', NO_ALBUMENTATIONS_UPDATE='1')
        subprocess.run([sys.executable, '-m', 'src.train_marker_context', '--help'], cwd=temp, env=env, check=True)
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                                 '-p', 'test_marker_context.py', '-v'], cwd=temp, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout.decode('utf-8', errors='replace'), flush=True)
        if result.returncode:
            raise RuntimeError('Fresh-directory integration checks failed; do not deliver bundle')
    summary = {'bundle': str(args.output.resolve()), 'files': len(FILES)+1,
               'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
               'standalone_cli_and_integration_checks': 'passed'}
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
