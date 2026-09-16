"""打包四个 marker 的推理结果为初赛提交 zip"""
import os
import zipfile
from pathlib import Path

ROOT = Path(r'E:\aic\ihc-virtual-stain')
RESULTS_ROOT = ROOT / 'results'
OUTPUT_ZIP = ROOT / 'submission.zip'

MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']

total_files = 0
with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for marker in MARKERS:
        marker_dir = RESULTS_ROOT / 'test' / marker
        if not marker_dir.is_dir():
            print(f"[SKIP] {marker}: directory not found")
            continue
        files = sorted(marker_dir.glob('*_fake.jpg'))
        print(f"[{marker}] adding {len(files)} files")
        for f in files:
            arcname = f"results/test/{marker}/{f.name}"
            zf.write(f, arcname)
            total_files += 1

print(f"\nZIP created: {OUTPUT_ZIP}")
print(f"Total files: {total_files}")
print(f"Size: {os.path.getsize(OUTPUT_ZIP) / 1024 / 1024:.2f} MB")
