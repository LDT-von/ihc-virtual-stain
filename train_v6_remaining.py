"""v6 训练剩余 3 marker (HLA-DR 已完成)
"""
import subprocess
from pathlib import Path

MARKERS = ["CD68", "CD45RO", "Vimentin"]
EPOCHS = 8
BATCH_SIZE = 16

CKPT_ROOT = Path("E:/aic/ihc-virtual-stain/checkpoints")


def is_done(marker: str) -> bool:
    for d in CKPT_ROOT.iterdir():
        if d.is_dir() and f"pix2pix_v6_{marker}" in d.name and (d / "best.pt").exists():
            return True
    return False


def train_one(marker: str):
    if is_done(marker):
        print(f"[SKIP] {marker} 已训练完成")
        return
    print(f"\n{'='*60}\n[TRAIN] {marker} starting...\n{'='*60}")
    log_path = Path(f"E:/aic/ihc-virtual-stain/train_v6_{marker}.log")
    cmd = [
        "python", "-u", "-m", "src.train_pyramid_pix2pix",
        "--marker", marker,
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--val_split", "0.05",
        "--log_every", "50",
    ]
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd="E:/aic/ihc-virtual-stain",
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        print(f"[FAIL] {marker} 训练失败，returncode={proc.returncode}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    for marker in MARKERS:
        train_one(marker)
        print(f"[PROGRESS] {marker} 完成，累计 {(time.time()-t0)/60:.1f} 分钟")
    print(f"\n[ALL DONE] 总用时 {(time.time()-t0)/60:.1f} 分钟")
