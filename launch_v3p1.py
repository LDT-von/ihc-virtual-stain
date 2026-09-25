# -*- coding: utf-8 -*-
"""v3p1 100-epoch training launcher.
Mirrors run_v3.py: subprocess.run with utf-8 list args."""
import subprocess, sys

data_root = r"E:\aic\复赛数据集(包括训练集和测试集输入)"
manifest = r"E:\aic\final-ihc\configs\roi_split_semifinal_2026_expanded.json"
init_ckpt = r"E:\aic\final-ihc\checkpoints\fullplus_cd68_v3\final.pt"
output = r"E:\aic\final-ihc\checkpoints\fullplus_cd68_v3p1"
script = r"E:\aic\final-ihc\train_fullplus_cd68.py"

cmd = [
    sys.executable, '-X', 'utf8', '-u', script,
    "--data-root", data_root,
    "--manifest", manifest,
    "--init-checkpoint", init_ckpt,
    "--output", output,
    "--epochs", "100",
    "--batch-size", "8",
    "--lr", "3e-5",
    "--cd68-weight", "3.0",
    "--seed", "2026",
]

print("Running:", " ".join(f'"{c}"' for c in cmd))
ret = subprocess.run(cmd)
sys.exit(ret.returncode)
