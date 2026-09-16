"""顺序训练所有 marker（除 HLA-DR 已续训中）"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(r'E:\aic\ihc-virtual-stain')
DATA_ROOT = r'E:\aic\ihc-virtual-stain\初赛数据集（包含训练集和测试集输入）\初赛数据集（包含训练集和测试集输入）'

# 训练配置
CONFIGS = [
    ('CD68', 30),
    ('CD45RO', 30),
    ('Vimentin', 30),
]


def train(marker: str, epochs: int):
    print(f"\n{'='*60}\n[TRAIN] {marker} for {epochs} epochs\n{'='*60}")
    cmd = [
        sys.executable, '-m', 'src.train_pix2pix_v2',
        '--marker', marker,
        '--epochs', str(epochs),
        '--batch_size', '24',
        '--lr', '2e-4',
        '--lambda_l1', '100',
        '--lambda_ssim', '50',
        '--num_workers', '2',
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[FAIL] {marker} training failed")
        sys.exit(1)


def main():
    for marker, epochs in CONFIGS:
        train(marker, epochs)
    print('\n[DONE] all training finished')


if __name__ == '__main__':
    main()
