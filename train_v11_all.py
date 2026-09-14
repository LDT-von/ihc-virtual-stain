"""v11 批量训练脚本：依次训练 4 个 marker

训练流程：
    HLA-DR → CD68 → CD45RO → Vimentin
    每个 marker: 20 epochs (val split) + 15 epochs (full-data 微调)

用法：
    python train_v11_all.py
"""
import subprocess
import time
import sys
from pathlib import Path

MARKERS = ["HLA-DR", "CD68", "CD45RO", "Vimentin"]

# 阶段1: 20 epochs 带 val split
EPOCHS_STAGE1 = 20
# 阶段2: 15 epochs 全量微调（基于 stage1 的 best）
EPOCHS_STAGE2 = 15
BATCH_SIZE = 16

CKPT_ROOT = Path("E:/aic/ihc-virtual-stain/checkpoints")
DATA_ROOT = "E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）"
WORK_DIR = Path("E:/aic/ihc-virtual-stain")


def is_done(marker: str) -> bool:
    """检查 marker 是否已完成两个阶段"""
    for d in CKPT_ROOT.iterdir():
        if d.is_dir() and f"pix2pix_v11_{marker}_stage2" in d.name:
            return (d / "best.pt").exists()
    return False


def get_stage1_dir(marker: str):
    for d in CKPT_ROOT.iterdir():
        if d.is_dir() and f"pix2pix_v11_{marker}_stage1" in d.name:
            return d
    return None


def run_training(marker: str, epochs: int, resume_ckpt: str = None,
                 suffix: str = "stage1", extra_args: list = None):
    log_file = WORK_DIR / f"train_v11_{marker}_{suffix}.log"
    args_list = [
        sys.executable, "-u", "-m", "src.train_pix2pix_v11",
        "--marker", marker,
        "--epochs", str(epochs),
        "--batch_size", str(BATCH_SIZE),
        "--data_root", DATA_ROOT,
    ]
    if resume_ckpt:
        args_list += ["--resume_ckpt", resume_ckpt]
        args_list += ["--full_data"]
    if extra_args:
        args_list += extra_args

    print(f"\n{'='*60}")
    print(f"[v11] {marker} ({suffix}): {epochs} epochs")
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
        print(f"[FAIL] {marker} {suffix} 失败，returncode={proc.returncode}")
        return False
    print(f"[OK] {marker} {suffix} 完成")
    return True


def main():
    t0 = time.time()

    for marker in MARKERS:
        print(f"\n{'#'*60}")
        print(f"# Marker: {marker}")
        print(f"# Time so far: {(time.time()-t0)/60:.1f} min")
        print(f"{'#'*60}")

        # ---- Stage 1: 20 epochs (val split, 找 best) ----
        s1_ok = run_training(
            marker,
            epochs=EPOCHS_STAGE1,
            suffix="stage1",
        )
        if not s1_ok:
            print(f"[WARN] {marker} stage1 失败，跳过")
            continue

        # 找到 stage1 的 best.pt
        s1_dir = get_stage1_dir(marker)
        if s1_dir is None:
            print(f"[ERROR] 找不到 {marker} stage1 checkpoint 目录")
            continue
        best_s1 = s1_dir / "best.pt"
        if not best_s1.exists():
            print(f"[WARN] {marker} stage1 无 best.pt，跳过 stage2")
            continue

        # ---- Stage 2: 15 epochs 全量微调 ----
        s2_ok = run_training(
            marker,
            epochs=EPOCHS_STAGE2,
            resume_ckpt=str(best_s1),
            suffix="stage2",
        )
        if not s2_ok:
            print(f"[WARN] {marker} stage2 失败")

        elapsed = (time.time() - t0) / 60
        print(f"\n[PROGRESS] {marker} 完成，累计 {elapsed:.1f} 分钟")

    total = (time.time() - t0) / 60
    print(f"\n{'#'*60}")
    print(f"[ALL DONE] 4 marker v11 训练总用时 {total:.1f} 分钟")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
