"""v6 一键推理 + 打包 - 扫描所有 v6 best.pt"""
import subprocess
import os
from pathlib import Path
import sys

CKPT_ROOT = Path("E:/aic/ihc-virtual-stain/checkpoints")
DATA_ROOT = "E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）"
RESULTS_DIR = "E:/aic/ihc-virtual-stain/results_v6"
MARKERS = ["HLA-DR", "CD68", "CD45RO", "Vimentin"]


def find_v6_ckpt(marker: str) -> Path:
    """找 v6 best.pt"""
    for d in sorted(CKPT_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir() and f"pix2pix_v6_{marker}" in d.name and (d / "best.pt").exists():
            return d / "best.pt"
    raise FileNotFoundError(f"找不到 {marker} 的 v6 best.pt")


def infer_marker(marker: str, ckpt: Path):
    cmd = [
        "python", "-u", "-m", "src.inference_pix2pix",
        "--ckpt", str(ckpt),
        "--marker", marker,
        "--data-root", DATA_ROOT,
        "--split", "test",
        "--output-dir", RESULTS_DIR,
        "--tta",
        "--jpeg-quality", "95",
    ]
    print(f"\n[INFER] {marker} <- {ckpt.name}")
    proc = subprocess.run(cmd, cwd="E:/aic/ihc-virtual-stain")
    return proc.returncode == 0


def package():
    cmd = [
        "python", "-m", "src.submit",
        "--results-dir", RESULTS_DIR,
        "--marker", "all",
        "--out-zip", "submission_v6.zip",
    ]
    # 实际打包：把所有 marker 都打包（修改 src.submit 支持 marker=all）
    print("\n[PACKAGE] 打包 v6 submission...")
    proc = subprocess.run(cmd, cwd="E:/aic/ihc-virtual-stain")


if __name__ == "__main__":
    # 1. 推理所有 marker
    for marker in MARKERS:
        try:
            ckpt = find_v6_ckpt(marker)
            ok = infer_marker(marker, ckpt)
            if not ok:
                print(f"[FAIL] {marker} 推理失败")
                sys.exit(1)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)

    # 2. 打包（手动合并，因为 src.submit 只支持单 marker）
    print("\n[PACKAGE] 手动打包 v6 submission.zip ...")
    import zipfile
    zip_path = Path("E:/aic/ihc-virtual-stain/submission_v6.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for marker in MARKERS:
            d = Path(RESULTS_DIR) / "test" / marker
            if not d.exists():
                print(f"[WARN] 缺少 {marker} 结果")
                continue
            for img in sorted(d.glob("*_fake.jpg")):
                arc = f"results/test/{marker}/{img.name}"
                zf.write(img, arc)
                print(f"  + {arc}")
    print(f"\n[OK] 打包完成: {zip_path}")
