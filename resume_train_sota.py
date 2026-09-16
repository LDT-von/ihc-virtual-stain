# -*- coding: utf-8 -*-
"""基于 SOTA 文献的继续微调：CSS Loss + LPIPS。

- 从每个 marker 最新的 checkpoint 加载
- 续训 N 个 epoch（默认 12，比纯继续更短，因为 CSS Loss 通常收敛更快）
- 使用 5% val：保留离线打分能力，避免无验证集盲打
- 关键技术：
    * --use_lpips：替换 VGG 为官方 LPIPS（感知更准）
    * --lambda_css 20.0：CSSP2P GAN 论文 (arXiv 2511.18946) 推荐的 CSS Loss 权重
    * --lambda_perceptual 5.0：LPIPS 权重（比 VGG 高更强）
    * lr=1e-5 极小：避免 catastrophic forgetting（CSS/LPIPS 引入新梯度方向）
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(r'E:\aic\ihc-virtual-stain')


def find_latest_ckpt(marker: str, prefer_best: bool = True):
    ck_root = ROOT / 'checkpoints'
    dirs = sorted(
        (d for d in ck_root.glob(f'pix2pix_v2_{marker}_*') if d.is_dir()),
        key=lambda d: int(d.name.rsplit('_', 1)[1]),
        reverse=True,
    )
    for d in dirs:
        if prefer_best and (d / 'best.pt').exists():
            return d / 'best.pt'
        epoch_files = sorted(d.glob('epoch*.pt'), key=lambda p: int(p.stem[5:]), reverse=True)
        if epoch_files:
            return epoch_files[0]
        if (d / 'final.pt').exists():
            return d / 'final.pt'
    return None


def train_one(marker: str, epochs: int, lr: float, resume_ckpt: Path,
              css: float, perc: float, edge: float, lpips: bool):
    cmd = [
        sys.executable, '-m', 'src.train_pix2pix_v2',
        '--marker', marker,
        '--epochs', str(epochs),
        '--batch_size', '24',
        '--lr', str(lr),
        '--lambda_l1', '100',
        '--lambda_ssim', '50',
        '--num_workers', '2',
        '--save_every', '2',
        '--val_split', '0.05',
        '--resume_ckpt', str(resume_ckpt),
        '--lambda_css', str(css),
        '--lambda_perceptual', str(perc),
        '--lambda_edge', str(edge),
        '--log_every', '50',
    ]
    if lpips:
        cmd.append('--use_lpips')
    print('\n' + '=' * 60)
    print('[Train]', marker, '| epochs=', epochs, '| lr=', lr)
    print('resume:', resume_ckpt)
    print('CSS=', css, 'LPIPS=', perc, 'EDGE=', edge, 'use_lpips=', lpips)
    print('cmd:', ' '.join(cmd))
    print('=' * 60)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f'[FAIL] {marker} 返回码 {result.returncode}')
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--markers', nargs='+',
                   default=['HLA-DR', 'CD68', 'CD45RO', 'Vimentin'])
    p.add_argument('--epochs', type=int, default=12,
                   help='每个 marker 续训 epoch 数')
    p.add_argument('--lr', type=float, default=1e-5,
                   help='极小学习率避免破坏已有知识')
    p.add_argument('--css', type=float, default=20.0,
                   help='CSS Loss 权重（论文推荐 10-25）')
    p.add_argument('--perceptual', type=float, default=5.0,
                   help='LPIPS 损失权重')
    p.add_argument('--edge', type=float, default=5.0)
    p.add_argument('--no_lpips', action='store_true')
    args = p.parse_args()

    for marker in args.markers:
        ckpt = find_latest_ckpt(marker, prefer_best=True)
        if ckpt is None:
            print(f'[Skip] {marker}: 无可用 checkpoint')
            continue
        ok = train_one(
            marker=marker,
            epochs=args.epochs,
            lr=args.lr,
            resume_ckpt=ckpt,
            css=args.css,
            perc=args.perceptual,
            edge=args.edge,
            lpips=not args.no_lpips,
        )
        if not ok:
            sys.exit(1)


if __name__ == '__main__':
    main()
