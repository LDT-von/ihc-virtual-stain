"""
v5: 全 best.pt + 4xTTA 推理
"""
import os
import subprocess
import shutil
from pathlib import Path

ROOT = Path(r'E:\aic\ihc-virtual-stain')
OUTROOT = ROOT / 'results_best_all_bestpt'
LOG = ROOT / 'infer_best_all_bestpt.log'

MARKERS = {
    'HLA-DR':   'pix2pix_v2_HLA-DR_1788962683\\best.pt',
    'CD68':     'pix2pix_v2_CD68_1788969904\\best.pt',
    'CD45RO':   'pix2pix_v2_CD45RO_1788978395\\best.pt',
    'Vimentin': 'pix2pix_v2_Vimentin_1788988426\\best.pt',
}

os.makedirs(OUTROOT, exist_ok=True)
os.makedirs(OUTROOT / 'test', exist_ok=True)

with open(LOG, 'w', encoding='utf-8') as f:
    f.write(f'=== v5: ALL best.pt + 4xTTA, started {os.environ.get("COMPUTERNAME","?")} ===\n\n')

for marker, ckpt_rel in MARKERS.items():
    ckpt_path = ROOT / 'checkpoints' / ckpt_rel
    out_dir = OUTROOT / 'test' / marker

    existing = len(list(out_dir.glob('*_fake.jpg'))) if out_dir.exists() else 0
    msg = f'[{marker}] ckpt={ckpt_path} existing={existing}'
    print(msg)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

    if existing >= 1346:
        print(f'  skip (done)')
        with open(LOG, 'a') as f:
            f.write(f'  skip (done)\n')
        continue

    if out_dir.exists():
        shutil.rmtree(out_dir)
        print(f'  removed incomplete dir')
        with open(LOG, 'a') as f:
            f.write('  removed incomplete dir\n')

    cmd = [
        'python',
        str(ROOT / 'inference_8x_tta.py'),
        '--marker', marker,
        '--ckpt', str(ckpt_path),
        '--split', 'test',
        '--tta-mode', '4x',
        '--batch-size', '4',
        '--output-root', str(OUTROOT),
    ]
    print(f'  launching: {" ".join(cmd)}')
    with open(LOG, 'a') as f:
        f.write(f'  launching...\n')
        f.flush()

    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(f'  stdout:\n{result.stdout}\n')
        if result.stderr:
            f.write(f'  stderr:\n{result.stderr}\n')

    done_count = len(list(out_dir.glob('*_fake.jpg'))) if out_dir.exists() else 0
    msg = f'  done: files={done_count}, exit={result.returncode}'
    print(msg)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')

with open(LOG, 'a') as f:
    f.write('\n=== v5 ALL DONE ===\n')
print('\nDone! Check infer_best_all_bestpt.log')
