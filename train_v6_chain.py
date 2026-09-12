"""v6 训练链：依次训练 4 marker (HLA-DR → CD68 → CD45RO → Vimentin)

每个 marker 完成后，自动启动下一个。
所有训练都使用 PyramidL1SSIMLoss 替换原损失。
"""
import subprocess
import time
from pathlib import Path

MARKERS = ["HLA-DR", "CD68", "CD45RO", "Vimentin"]
EPOCHS = 20
BATCH_SIZE = 16

# 已完成的 marker（基于 best.pt 存在判断）
CKPT_ROOT = Path("E:/aic/ihc-virtual-stain/checkpoints")


def is_done(marker: str) -> bool:
    """检查 marker 是否已训练完成（任何 v6 best.pt 存在）"""
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
        "--full_data",
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
        raise RuntimeError(f"Training failed for {marker}")
    print(f"[DONE] {marker} 训练完成")


if __name__ == "__main__":
    t0 = time.time()
    for marker in MARKERS:
        train_one(marker)
        elapsed = (time.time() - t0) / 60
        print(f"[PROGRESS] 已用时 {elapsed:.1f} 分钟")
    print(f"\n[ALL DONE] 4 marker 训练总用时 {(time.time()-t0)/60:.1f} 分钟")
