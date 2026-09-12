"""列出所有 v6 checkpoint 的 best.pt"""
from pathlib import Path

for d in Path("E:/aic/ihc-virtual-stain/checkpoints").iterdir():
    if "pix2pix_v6" in d.name and d.is_dir():
        best = d / "best.pt"
        if best.exists():
            print(f"{best}  size={best.stat().st_size/1024/1024:.1f}MB")
