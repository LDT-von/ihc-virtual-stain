"""从各 marker 最新 checkpoint 续训更长的 epoch。

策略：
- 对每个 marker 找到最新 pix2pix_v2_<marker>_* 目录里带优化器状态的最高 epoch checkpoint
- 续训 CONTINUE_EPOCHS 轮，训练过程中按验证集 SSIM 自动保存 best.pt（train_pix2pix_v2 内置）
- 每个 epoch 的 val SSIM/PSNR 都会写入新 run 的 train_log.jsonl，用于训练后挑选最优
"""
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(r'E:\aic\ihc-virtual-stain')
MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']
CONTINUE_EPOCHS = 50  # 每 marker 额外续训轮数


def find_resume_ckpt(marker: str) -> Path | None:
    """在最新 run 目录里挑带优化器状态的最高 epoch checkpoint。"""
    ck_root = ROOT / 'checkpoints'
    dirs = sorted(
        (d for d in ck_root.glob(f'pix2pix_v2_{marker}_*') if d.is_dir()),
        key=lambda d: int(d.name.rsplit('_', 1)[1]),
        reverse=True,
    )
    for newest in dirs:
        # 跳过空目录/无 checkpoint 的残留目录
        epoch_files = list(newest.glob('epoch*.pt'))
        if not epoch_files and not (newest / 'final.pt').exists():
            continue

        def epoch_num(name: str) -> int:
            return int(name[len('epoch'):-len('.pt')])

        epoch_files = sorted(epoch_files, key=lambda p: epoch_num(p.name), reverse=True)
        for p in epoch_files:
            try:
                ckpt = torch.load(p, map_location='cpu', weights_only=False)
            except Exception as e:  # 文件损坏则跳过
                print(f'[Skip] 无法读取 {p}: {e}')
                continue
            if 'optimizer_g' in ckpt:
                return p
        # 没有带优化器状态的 ckpt，退回 final.pt（仅模型，优化器重启）
        final = newest / 'final.pt'
        if final.exists():
            print(f'[Warn] {marker} 未找到带优化器的 epoch ckpt，退回 {final}')
            return final
    return None


def train(marker: str, epochs: int, resume_ckpt: Path):
    print(f"\n{'=' * 60}\n[TRAIN] {marker}: 续训 {epochs} epochs from {resume_ckpt.name}\n{'=' * 60}", flush=True)
    cmd = [
        sys.executable, '-m', 'src.train_pix2pix_v2',
        '--marker', marker,
        '--epochs', str(epochs),
        '--batch_size', '24',
        '--lr', '2e-4',
        '--lambda_l1', '100',
        '--lambda_ssim', '50',
        '--num_workers', '2',
        '--save_every', '5',
        '--resume_ckpt', str(resume_ckpt),
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[FAIL] {marker} 续训失败")
        sys.exit(1)


def main():
    for marker in MARKERS:
        resume_ckpt = find_resume_ckpt(marker)
        if resume_ckpt is None:
            print(f"[SKIP] {marker} 无可用 checkpoint")
            continue
        train(marker, CONTINUE_EPOCHS, resume_ckpt)
    print('\n[DONE] all resumed training finished')


if __name__ == '__main__':
    main()
