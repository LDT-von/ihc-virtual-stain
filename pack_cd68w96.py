"""Pack cd68_weighted_w96/best.pt → submission zip.

avg_ssim=0.842 (CD68 SSIM 0.82, others ~0.85).
Candidate to beat v3 (75.1128)."""
import sys, time, shutil, subprocess
from pathlib import Path

ROOT = Path(r'E:\aic\final-ihc')
CKPT = ROOT / 'checkpoints/cd68_weighted_w96/best.pt'
TEST_INPUT = ROOT.parent / '复赛数据集(包括训练集和测试集输入)/test/DAPI'
OUT_DIR = ROOT / 'results_cd68w96'
ZIP_OUT = ROOT / 'submissions/cd68_weighted_w96_rgb_FIXED.zip'

MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']

print(f'=== Pack cd68_weighted_w96 ===')
print(f'  ckpt: {CKPT}')
print(f'  test: {TEST_INPUT}')
print(f'  out : {OUT_DIR}')
print(f'  zip : {ZIP_OUT}')

if not CKPT.exists():
    sys.exit(f'ERROR: {CKPT} not found')

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)

# Step 1: inference
print(f'\nStep 1: inference')
cmd = [sys.executable, '-X', 'utf8', '-u', '-m', 'src.train_marker_context', 'infer',
       '--input', str(TEST_INPUT),
       '--checkpoint', str(CKPT),
       '--output', str(OUT_DIR),
       '--batch-size', '32',
       '--device', 'cuda',
       ]
ret = subprocess.run(cmd, cwd=str(ROOT))
if ret.returncode != 0:
    sys.exit(f'inference failed: {ret.returncode}')

# Step 2: zip
print(f'\nStep 2: zip')
import zipfile
ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)
if ZIP_OUT.exists():
    ZIP_OUT.unlink()

total = 0
total_bytes = 0
src_dir = OUT_DIR / 'results' / 'test'
with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for marker in MARKERS:
        marker_dir = src_dir / marker
        jpegs = sorted(marker_dir.glob('*_fake.jpg'))
        print(f'  {marker}: {len(jpegs)} JPEGs')
        for j in jpegs:
            arcname = f'results/test/{marker}/{j.name}'
            data = j.read_bytes()
            total_bytes += len(data)
            zf.writestr(arcname, data)
            total += 1

size_mb = ZIP_OUT.stat().st_size / 1e6
print(f'\nDone: {total} files -> {ZIP_OUT} ({size_mb:.1f} MB)')
print(f'Avg file: {total_bytes/total/1024:.1f} KB')
print(f'\n=== Submit {ZIP_OUT.name} on platform ===')
