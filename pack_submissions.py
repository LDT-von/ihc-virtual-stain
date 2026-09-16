"""打包四个 marker 的推理结果为初赛提交 zip。

ZIP 内路径布局（与 submission_v12_bestv2.zip 一致）：
    test/<marker>/<name>_fake.jpg

ZIP 文件名: submission_v<VERSION>.zip （VERSION 文件控制版本号）。
"""
import zipfile
from pathlib import Path

ROOT = Path('.').resolve()
RESULTS_ROOT = ROOT / 'results' / 'test'
MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']

version_file = ROOT / 'VERSION'
version = version_file.read_text().strip() if version_file.exists() else '1'
out_zip = ROOT / f'submission_v{version}.zip'

total = 0
with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for marker in MARKERS:
        marker_dir = RESULTS_ROOT / marker
        if not marker_dir.is_dir():
            print(f"[SKIP] {marker}: not found under {marker_dir}")
            continue
        files = sorted(marker_dir.glob('*_fake.jpg'))
        print(f"[{marker}] adding {len(files)} files")
        for f in files:
            arcname = f"test/{marker}/{f.name}"
            zf.write(f, arcname)
            total += 1

print(f"\nZIP: {out_zip}")
print(f"Total files: {total}")
print(f"Size: {out_zip.stat().st_size/1024/1024:.2f} MB")
