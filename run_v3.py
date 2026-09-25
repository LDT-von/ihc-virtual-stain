# -*- coding: utf-8 -*-
import subprocess
import sys

data_root = r"E:\aic\复赛数据集(包括训练集和测试集输入)"
manifest = r"E:\aic\final-ihc\configs\roi_split_semifinal_2026_expanded.json"
init_ckpt = r"E:\aic\final-ihc\checkpoints\ultimate_w96_expanded\baseline\best.pt"
output = r"E:\aic\final-ihc\checkpoints\fullplus_cd68_v3"

script = r"E:\aic\final-ihc\train_fullplus_cd68.py"
python = r"D:\Anaconda3\python.exe"

cmd = [
    python, script,
    "--data-root", data_root,
    "--manifest", manifest,
    "--init-checkpoint", init_ckpt,
    "--output", output,
    "--epochs", "50",
    "--batch-size", "8",
    "--lr", "3e-5",
    "--cd68-weight", "3.0",
]

print("Running:", " ".join(f'"{c}"' for c in cmd))
subprocess.run(cmd)
