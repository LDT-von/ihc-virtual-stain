"""Train W=128 (larger capacity) for ultimate performance."""
import subprocess
import sys
from pathlib import Path

# Paths
FINAL_IHC = Path(r"E:\aic\final-ihc")
CKPT_DIR = FINAL_IHC / "checkpoints" / "ultimate_w128_expanded"
DATA_ROOT = Path(r"E:\aic\复赛数据集(包括训练集和测试集输入)")
CONFIG = FINAL_IHC / "configs" / "roi_split_semifinal_2026_expanded.json"
PYTHON = r"D:\Anaconda3\python.exe"

CKPT_DIR.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-m", "src.train_marker_context",
    "--config", str(CONFIG),
    "--width", "128",
    "--epochs", "30",
    "--batch-size", "16",  # Reduced due to larger model
    "--patch-size", "256",
    "--lr", "2e-4",
    "--output", str(CKPT_DIR),
    "--device", "cuda",
]

print(f"Command: {' '.join(cmd)}")
print(f"Output dir: {CKPT_DIR}")
print(f"Expected time: 1-2 hours")
print("-" * 60)

result = subprocess.run(cmd, cwd=str(FINAL_IHC))
sys.exit(result.returncode)
