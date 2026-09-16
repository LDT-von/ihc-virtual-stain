import os
from pathlib import Path

results_dir = Path("E:/aic/ihc-virtual-stain/results")
if results_dir.exists():
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            fpath = Path(root) / f
            print(f"{fpath} ({os.path.getsize(fpath) / 1024 / 1024:.2f} MB)")
else:
    print("Results folder does not exist")
