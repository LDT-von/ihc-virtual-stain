"""将 8xTTA test 结果打包为 submission_v2.zip
- test/HLA-DR/<name>_fake.jpg
- test/CD68/<name>_fake.jpg
- test/CD45RO/<name>_fake.jpg
- test/Vimentin/<name>_fake.jpg
"""
import os
import zipfile
from pathlib import Path

ROOT = Path(r'E:\aic\ihc-virtual-stain')
RESULTS_ROOT = ROOT / 'results_8xtta_test'
OUTPUT_ZIP = ROOT / 'submission_v2.zip'

MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']

total_files = 0
with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for marker in MARKERS:
        marker_dir = RESULTS_ROOT / 'test' / marker
        if not marker_dir.is_dir():
            print(f'[SKIP] {marker}: directory not found at {marker_dir}')
            continue
        files = sorted(marker_dir.glob('*_fake.jpg'))
        print(f'[{marker}] adding {len(files)} files')
        for f in files:
            arcname = f'test/{marker}/{f.name}'
            zf.write(f, arcname)
            total_files += 1

print(f'\nZIP created: {OUTPUT_ZIP}')
print(f'Total files: {total_files}')
print(f'Size: {os.path.getsize(OUTPUT_ZIP) / 1024 / 1024:.2f} MB')
