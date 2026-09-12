"""打包 v10 测试集结果（含 TTA + ensemble）

v10 改进：
- 4-flip TTA
- Vimentin 3-checkpoint ensemble
- Val SSIM: CD45RO 0.7031->0.7042, Vimentin 0.7515->0.7523,
            HLA-DR 0.7700->0.7713, CD68 0.7976->0.8022
"""
import os
import zipfile
from pathlib import Path

ROOT = Path(r'E:\aic\ihc-virtual-stain')
RESULTS_ROOT = ROOT / 'results_v10_test' / 'test'
OUTPUT_ZIP = ROOT / 'submission_v10.zip'

MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']

# 4-flip TTA 后的 val SSIM（从 test_tta_val.py 测得）
SCORES = {
    'HLA-DR': 0.7713,
    'CD68': 0.8022,
    'CD45RO': 0.7042,
    'Vimentin': 0.7523,  # ensemble 3 + TTA
}

total_files = 0
with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for marker in MARKERS:
        marker_dir = RESULTS_ROOT / marker
        if not marker_dir.is_dir():
            print(f'[SKIP] {marker}: directory not found')
            continue
        files = sorted(marker_dir.glob('*_fake.jpg'))
        print(f'[{marker}] adding {len(files)} files (val_ssim={SCORES[marker]:.4f})')
        for f in files:
            arcname = f'test/{marker}/{f.name}'
            zf.write(f, arcname)
            total_files += 1

print(f'\nZIP created: {OUTPUT_ZIP}')
print(f'Total files: {total_files}')
print(f'Size: {os.path.getsize(OUTPUT_ZIP) / 1024 / 1024:.2f} MB')

avg_ssim = sum(SCORES.values()) / len(SCORES)
print(f'\n=== v10 Submission Summary ===')
for marker, ssim in SCORES.items():
    print(f'  {marker}: SSIM={ssim:.4f}')
print(f'  Average SSIM: {avg_ssim:.4f}')
print(f'  Estimated platform score: {avg_ssim * 100:.2f}')
