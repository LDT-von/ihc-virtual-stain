"""顺序训练所有 4 个 marker 的 NAFNet"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(r'E:\aic\ihc-virtual-stain')
DATA_ROOT = r'E:\aic\ihc-virtual-stain\初赛数据集（包含训练集和测试集输入）\初赛数据集（包含训练集和测试集输入）'

# 训练配置 - 4个marker各20轮
CONFIGS = [
    ('CD68', 20),
    ('CD45RO', 20),
    ('Vimentin', 20),
]


def train(marker: str, epochs: int):
    print(f"\n{'='*60}\n[NAFNet] {marker} for {epochs} epochs\n{'='*60}")
    cmd = [
        sys.executable,
        str(ROOT / 'train_nafnet.py'),
        '--marker', marker,
        '--epochs', str(epochs),
        '--batch-size', '16',
        '--lr', '2e-4',
        '--width', '64',
        '--save-name', 'nafnet_full',
        '--data-root', DATA_ROOT,
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[FAIL] {marker} NAFNet training failed with code {result.returncode}")
        return False
    print(f"[OK] {marker} NAFNet training completed")
    return True


def main():
    for marker, epochs in CONFIGS:
        success = train(marker, epochs)
        if not success:
            print(f"[ERROR] Stopping - {marker} failed")
            sys.exit(1)
    print('\n[DONE] All 4 marker NAFNet training finished!')
    print('Models saved in: checkpoints/nafnet_full_<marker>_<timestamp>/')


if __name__ == '__main__':
    main()
