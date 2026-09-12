"""打包 v7 测试集结果为 submission_v7.zip

results/v7/test/<marker>/<name>_fake.jpg -->

submission_v7.zip
  test/<marker>/<name>_fake.jpg
"""
import os
import zipfile
from pathlib import Path

ROOT = Path(r'E:\aic\ihc-virtual-stain')
RESULTS_ROOT = ROOT / 'results_v7_test' / 'test'
OUTPUT_ZIP = ROOT / 'submission_v7.zip'

MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']

# 验证集分数
SCORES = {
    'HLA-DR': 0.4005,
    'CD68': 0.8190,
    'CD45RO': 0.5877,
    'Vimentin': 0.5844,
}

total_files = 0
with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for marker in MARKERS:
        marker_dir = RESULTS_ROOT / marker
        if not marker_dir.is_dir():
            print(f'[SKIP] {marker}: directory not found at {marker_dir}')
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
print(f'\n=== v7 Submission Summary ===')
for marker, ssim in SCORES.items():
    print(f'  {marker}: SSIM={ssim:.4f}')
print(f'  Average SSIM: {avg_ssim:.4f}')
print(f'  Platform score: {avg_ssim * 100:.2f}')
