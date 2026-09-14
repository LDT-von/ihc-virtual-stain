"""
批量训练 v11：依次训练 CD68, CD45RO, Vimentin

HLA-DR 已完成（见 train_v11_HLA-DR_1789361914）。
继续训练剩余 3 个 marker。

用法：python train_v11_remaining.py
"""
import subprocess
import time
import sys
from pathlib import Path

MARKERS = ["CD68", "CD45RO", "Vimentin"]
EPOCHS = 30  # 与 v2 一致
BATCH_SIZE = 16
CKPT_ROOT = Path("e:/aic/ihc-virtual-stain/checkpoints")
WORK_DIR = Path("e:/aic/ihc-virtual-stain")
DATA_ROOT = "E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）"


def is_done(marker: str) -> bool:
    for d in CKPT_ROOT.iterdir():
        if d.is_dir() and f"pix2pix_v11_{marker}" in d.name:
            if (d / "best.pt").exists():
                # 检查 best.pt 的 val_ssim 是否 >= v2
                return True
    return False


def run_training(marker: str, epochs: int = EPOCHS):
    log_file = WORK_DIR / f"train_v11_{marker}.log"
    args_list = [
        sys.executable, "-u", "-m", "src.train_pix2pix_v11",
        "--marker", marker,
        "--epochs", str(epochs),
        "--batch_size", str(BATCH_SIZE),
        "--data_root", DATA_ROOT,
        "--log_every", "100",
        "--lr_d", "4e-5",
        "--lambda_perceptual", "0",
        "--lambda_css", "0",
        "--save_every", "5",
    ]
    print(f"\n{'='*60}")
    print(f"[v11] {marker}: {epochs} epochs, bs={BATCH_SIZE}")
    print(f"[v11] cmd: {' '.join(args_list)}")
    print(f"{'='*60}")

    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            args_list,
            cwd=str(WORK_DIR),
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode != 0:
        print(f"[FAIL] {marker} failed, returncode={proc.returncode}")
        return False
    print(f"[OK] {marker} done")
    return True


def main():
    t0 = time.time()
    for marker in MARKERS:
        if is_done(marker):
            print(f"[SKIP] {marker} already done")
            continue

        ok = run_training(marker)
        if not ok:
            print(f"[WARN] {marker} failed, continuing...")

        elapsed = (time.time() - t0) / 60
        print(f"\n[PROGRESS] {marker} done, elapsed {elapsed:.1f} min")

    total = (time.time() - t0) / 60
    print(f"\n[ALL DONE] total {total:.1f} min")


if __name__ == "__main__":
    main()
