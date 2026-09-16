"""Exercise the public training/inference/submission entry points with real data.

The tiny temporary checkpoint is only a plumbing fixture. No image-quality or
ensemble-gain claims may be made from its outputs. Official test images are not
used: all fixtures come from the published training set and disjoint ROI IDs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
import yaml

from src.models.flow_matching import FlowMatching, build_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT /
                        '初赛数据集（包含训练集和测试集输入）' /
                        '初赛数据集（包含训练集和测试集输入）')
    parser.add_argument('--marker', default='CD68')
    parser.add_argument('--report', type=Path, default=ROOT / 'tmp' /
                        'integration_test_report.json')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    report = {'purpose': 'pipeline integration; not model quality evaluation',
              'marker': args.marker, 'official_test_used': False, 'checks': []}
    start = time.perf_counter()

    def run_cli(module: str, *arguments: str) -> subprocess.CompletedProcess:
        # Configure UTF-8 after Python startup: this host has a legacy-encoded
        # site .pth file that prevents starting Python with -X utf8.
        bootstrap = ("import sys,runpy; sys.stdout.reconfigure(encoding='utf-8'); "
                     "sys.stderr.reconfigure(encoding='utf-8'); "
                     f"runpy.run_module('{module}',run_name='__main__')")
        env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE='1')
        return subprocess.run([sys.executable, '-c', bootstrap, *map(str, arguments)],
                              cwd=ROOT, env=env, capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=180)

    def check(name, fn):
        before = time.perf_counter()
        try:
            detail = fn()
            row = {'name': name, 'passed': True, 'detail': detail}
        except Exception as exc:
            row = {'name': name, 'passed': False,
                   'detail': f'{type(exc).__name__}: {exc}'}
        row['seconds'] = round(time.perf_counter() - before, 3)
        report['checks'].append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'} {name}: {row['detail']}", flush=True)

    # Every temporary file is created inside this one newly allocated directory.
    # Existing project datasets/checkpoints/configs are never overwritten.
    (ROOT / 'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pipeline-', dir=ROOT / 'tmp') as temp:
        fixture = Path(temp)
        candidates = sorted((args.data_root / 'train' / 'DAPI').glob('*.jpg'))
        first = candidates[0]
        second = next(p for p in candidates if p.name.split('_')[0] != first.name.split('_')[0])
        report['fixture_names'] = [first.name, second.name]
        report['fixture_note'] = 'One optimizer step; 256x256; disjoint ROI fixture only.'
        for split, source in [('train', first), ('val', second), ('test', second)]:
            for marker in ['DAPI'] + ([args.marker] if split != 'test' else []):
                destination = fixture / 'data' / split / marker / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(args.data_root / 'train' / marker / source.name, destination)

        cfg = yaml.safe_load((ROOT / 'configs' / 'default.yaml').read_text(encoding='utf-8'))
        cfg['data'].update(root=str(fixture / 'data'), batch_size=1, num_workers=0, augment=False)
        cfg['defaults'].update(marker=args.marker, amp=False)
        cfg['train'].update(epochs=1, log_every=1, save_every=1)
        cfg['model'].update(base_channels=32, channel_mults=[1, 2, 4],
                            num_res_blocks=1, attention_resolutions=[], dropout=0.0)
        cfg['flow_matching'].update(num_sampling_steps=2, solver='euler')
        config_path = fixture / 'config.yaml'
        config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')

        # Redirect relative CLI outputs into this fixture by giving each child
        # its own working directory, with imports resolved to the actual source.
        original_cli = run_cli

        def run_cli(module: str, *arguments: str) -> subprocess.CompletedProcess:
            bootstrap = ("import sys,runpy; sys.stdout.reconfigure(encoding='utf-8'); "
                         "sys.stderr.reconfigure(encoding='utf-8'); "
                         f"sys.path.insert(0,{str(ROOT)!r}); "
                         f"runpy.run_module('{module}',run_name='__main__')")
            return subprocess.run([sys.executable, '-c', bootstrap, *map(str, arguments)],
                                  cwd=fixture, env=dict(os.environ, NO_ALBUMENTATIONS_UPDATE='1'),
                                  capture_output=True, text=True, encoding='utf-8',
                                  errors='replace', timeout=180)

        def train_cli():
            result = run_cli('src.train', '--config', config_path)
            if result.returncode:
                raise RuntimeError(result.stderr.strip().splitlines()[-1])
            assert list((fixture / 'checkpoints').rglob('final.pt')), 'Final checkpoint missing'
            return 'One real training pair -> optimizer update -> final checkpoint'

        check('training_cli_to_checkpoint', train_cli)

        checkpoint = fixture / 'smoke_only.pt'

        def checkpoint_roundtrip():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            kwargs = {k: cfg['model'][k] for k in ['in_channels', 'cond_channels',
                      'base_channels', 'channel_mults', 'num_res_blocks',
                      'attention_resolutions', 'dropout']}
            model = build_model(**kwargs).to(device)
            fm = FlowMatching(model)
            def tensor(marker):
                with Image.open(fixture / 'data' / 'train' / marker / first.name) as im:
                    a = np.array(im.convert('RGB'), copy=True)
                return torch.from_numpy(a).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1
            dapi, target = tensor('DAPI'), tensor(args.marker)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            loss, _, _ = fm.forward_train(target, dapi)
            assert torch.isfinite(loss), 'Non-finite training loss'
            loss.backward()
            optimizer.step()
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({'model': state, 'cfg': cfg, 'smoke_only': True}, checkpoint)
            restored = build_model(**kwargs)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            restored.load_state_dict(saved['model'])
            assert all(torch.equal(value, restored.state_dict()[key]) for key, value in state.items())
            return 'Real 256x256 pair; one update; all saved/restored parameters identical'

        check('real_pair_checkpoint_roundtrip', checkpoint_roundtrip)
        torch.cuda.empty_cache()

        def validation_cli():
            result = run_cli('src.inference', '--ckpt', checkpoint, '--split', 'val',
                             '--output_dir', fixture / 'val_outputs', '--save_images')
            if result.returncode:
                raise RuntimeError(result.stderr.strip().splitlines()[-1])
            paths = list((fixture / 'val_outputs' / 'images').glob('*.png'))
            assert len(paths) == 1, f'Expected one output; got {len(paths)}'
            with Image.open(paths[0]) as image:
                assert image.size == (256, 256), f'Wrong output size: {image.size}'
            assert 'n=1' in result.stdout, 'Validation count was not one'
            return 'CLI reload -> image generation -> 256x256 output -> metrics against real target'

        check('validation_inference_cli', validation_cli)

        def unlabelled_cli():
            result = run_cli('src.inference', '--ckpt', checkpoint, '--split', 'test',
                             '--output_dir', fixture / 'test_outputs', '--save_images')
            if result.returncode:
                raise RuntimeError(result.stderr.strip().splitlines()[-1])
            assert 'SSIM=' not in result.stdout and 'PSNR=' not in result.stdout, (
                'Unlabelled test split incorrectly reports SSIM/PSNR against placeholder targets')
            return 'Unlabelled inference produces images without fabricated reference metrics'

        check('unlabelled_inference_metric_contract', unlabelled_cli)

        def submission_cli():
            # Run packaging from the real repo so src/ and configs/ really exist.
            archive = fixture / 'submission.zip'
            result = original_cli('src.submit', '--ckpt', checkpoint, '--marker', args.marker,
                                  '--output_dir', fixture / 'val_outputs', '--out_zip', archive)
            if result.returncode:
                raise RuntimeError(result.stderr.strip().splitlines()[-1])
            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
            assert any(n.startswith('src/') and n.endswith('.py') for n in names), 'Archive missing source code'
            assert any(n.startswith('configs/') and n.endswith('.yaml') for n in names), 'Archive missing configuration'
            assert any(n.endswith('.pt') for n in names), 'Archive missing model checkpoint'
            assert any(n.endswith('.png') for n in names), 'Archive missing predictions'
            return 'Submission archive contains images, source, configuration, and checkpoint'

        check('submission_archive_contract', submission_cli)

    report['seconds'] = round(time.perf_counter() - start, 3)
    report['passed'] = sum(row['passed'] for row in report['checks'])
    report['failed'] = len(report['checks']) - report['passed']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Report: {args.report}; {report['passed']} passed, {report['failed']} failed", flush=True)
    return int(report['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
