"""顺序调度：在 CD68 完成后依次训练 CD45RO、Vimentin"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r'E:\aic\ihc-virtual-stain')

CONFIGS = [
    ('CD45RO', 30),
    ('Vimentin', 30),
]


def train(marker: str, epochs: int):
    print(f"\n{'='*60}\n[TRAIN] {marker} for {epochs} epochs\n{'='*60}", flush=True)
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


def wait_for_cd68_done():
    """等待 CD68 的 final.pt 出现"""
    print("[WAIT] waiting for CD68 final.pt ...", flush=True)
    while True:
        ck_root = ROOT / 'checkpoints'
        for d in ck_root.iterdir():
            if d.name.startswith('pix2pix_v2_CD68'):
                if (d / 'final.pt').exists():
                    print(f"[OK] CD68 done: {d}", flush=True)
                    return
        time.sleep(30)


def main():
    wait_for_cd68_done()
    for marker, epochs in CONFIGS:
        train(marker, epochs)
    print('\n[DONE] all training finished')


if __name__ == '__main__':
    main()
