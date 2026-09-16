"""用 67.4586 那组 ckpt 跑 4xTTA test 推理
- HLA-DR:    pix2pix_v2_HLA-DR_1788962683/epoch89.pt  (val 0.78)
- CD68:      pix2pix_v2_CD68_1788969904/...
- CD45RO:    pix2pix_v2_CD45RO_1788978395/...
- Vimentin:  pix2pix_v2_Vimentin_1788988426/epoch69.pt
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(r'E:\aic\ihc-virtual-stain')
CKPT_BASE = ROOT / 'checkpoints'
OUTPUT_ROOT = ROOT / 'results_best_4xtta'

# 67.4586 那一组的 ckpt
BEST_CKPTS = {
    'HLA-DR':   CKPT_BASE / 'pix2pix_v2_HLA-DR_1788962683' / 'epoch89.pt',
    'CD68':     CKPT_BASE / 'pix2pix_v2_CD68_1788969904' / 'best.pt',
    'CD45RO':   CKPT_BASE / 'pix2pix_v2_CD45RO_1788978395' / 'best.pt',
    'Vimentin': CKPT_BASE / 'pix2pix_v2_Vimentin_1788988426' / 'epoch69.pt',
}

# 先确认 ckpt 存在
for marker, ckpt in BEST_CKPTS.items():
    if not ckpt.exists():
        print(f'[MISS] {marker}: {ckpt} NOT FOUND')
        # 列出该目录下的所有 epoch
        ckpt_dir = ckpt.parent
        if ckpt_dir.exists():
            print(f'  Available:')
            for f in sorted(ckpt_dir.glob('*.pt')):
                print(f'    {f.name}')
        sys.exit(1)
    else:
        print(f'[OK] {marker}: {ckpt.name}')

# 对每个 marker 跑 4xTTA test 推理
for marker, ckpt in BEST_CKPTS.items():
    out_dir = OUTPUT_ROOT / 'test' / marker
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'python', 'inference_8x_tta.py',
        '--marker', marker,
        '--ckpt', str(ckpt),
        '--batch-size', '4',
        '--split', 'test',
        '--tta-mode', '4x',  # 4xTTA 跟历史最佳一致
        '--output-dir', str(out_dir),
    ]
    print(f'\n[RUN] {marker} 4xTTA')
    print(f'  cmd: {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f'[FAIL] {marker}')
        sys.exit(1)

print('\n[DONE] All markers 4xTTA test inference completed')
print(f'Results dir: {OUTPUT_ROOT}')
