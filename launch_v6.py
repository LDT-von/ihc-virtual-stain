# -*- coding: utf-8
"""Launch v6: scratch, 300 epoch, progressive aug weaken from epoch 200."""
import subprocess, sys

data_root = r"E:\aic\复赛数据集(包括训练集和测试集输入)"
manifest = r"E:\aic\final-ihc\configs\roi_split_semifinal_2026_expanded.json"
output = r"E:\aic\final-ihc\checkpoints\fullplus_cd68_v6"
script = r"E:\aic\final-ihc\train_v6.py"

cmd = [
    sys.executable, '-X', 'utf8', '-u', script,
    '--data-root', data_root,
    '--manifest', manifest,
    '--output', output,
    '--epochs', '300',
    '--save-every', '50',
    '--batch-size', '8',
    '--lr', '3e-4',
    '--warmup-epochs', '10',
    '--weaken-start-epoch', '200',
    '--cd68-weight', '3.0',
    '--width', '96',
    '--seed', '2026',
]
print("Running:", " ".join(f'"{c}"' for c in cmd))
ret = subprocess.run(cmd, cwd=r'E:\aic\final-ihc')
sys.exit(ret.returncode)
