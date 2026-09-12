"""打包 v8 测试集结果为 submission_v8.zip

使用最佳 epoch / ensemble:
- HLA-DR: best.pt (0.7700)
- CD68: best.pt (0.7976)
- CD45RO: epoch57 (0.7031)
- Vimentin: epoch58+60+62 ensemble (0.7515)
"""
import os
import zipfile
from pathlib import Path

ROOT = Path(r'E:\aic\ihc-virtual-stain')
RESULTS_ROOT = ROOT / 'results_v8_test' / 'test'
OUTPUT_ZIP = ROOT / 'submission_v8.zip'

MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']

SCORES = {
    'HLA-DR': 0.7700,
    'CD68': 0.7976,
    'CD45RO': 0.7031,
    'Vimentin': 0.7515,
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
print(f'\n=== v8 Submission Summary ===')
for marker, ssim in SCORES.items():
    print(f'  {marker}: SSIM={ssim:.4f}')
print(f'  Average SSIM: {avg_ssim:.4f}')
print(f'  Estimated platform score: {avg_ssim * 100:.2f}')
