import zipfile
from pathlib import Path

zip_file = Path("E:/aic/ihc-virtual-stain/results_test_pix2pix.zip")
with zipfile.ZipFile(zip_file, 'r') as zf:
    names = zf.namelist()
    print(f"ZIP contents: {len(names)} files")
    print("First 5 files:")
    for name in sorted(names)[:5]:
        print(f"  {name}")
    print("...")
    print("Last 5 files:")
    for name in sorted(names)[-5:]:
        print(f"  {name}")
