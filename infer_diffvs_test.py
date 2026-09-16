"""对测试集 DAPI 跑 Stage1 推理，直接回归 (无 FM)。
支持多个 marker，逐一生成 results/test/<marker>/*.jpg
"""
import argparse
import time
from pathlib import Path
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader

# 导入训练脚本中的模型
sys.path.insert(0, str(Path(__file__).parent))
from train_diffvs_latent_fm import DiffVSFull
from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import MetricAggregator, to_uint8

MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--marker', required=True, choices=MARKERS)
    p.add_argument('--data-root', default='E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
    p.add_argument('--output-dir', default='results')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--jpeg-quality', type=int, default=95)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Infer] marker={args.marker} ckpt={args.ckpt}')
    print(f'[Infer] device={device}')

    # 构建模型（必须与训练一致：base_ch=64, marker_dim=32）
    model = DiffVSFull(base_ch=64, num_markers=4, marker_dim=32).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'], strict=False)
    model.eval()
    print(f'[Infer] loaded ckpt from epoch {ck.get("epoch")}, val_ssim={ck.get("ssim", -1):.4f}')

    # 数据集
    data_root = Path(args.data_root)
    marker_idx = MARKERS.index(args.marker)
    ds = DAPItoIHCDataset(data_root, args.marker, split='test', patch_size=256, augment=False)
    print(f'[Infer] test set size: {len(ds)}')

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_batch(b, marker_idx)
    )

    output_dir = Path(args.output_dir) / 'test' / args.marker
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    n_written = 0
    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device)
            midx = batch['marker_idx'].to(device)
            pred = model.forward_stage1(dapi, midx)
            for i, name in enumerate(batch['name']):
                img = Image.fromarray(to_uint8(pred[i].cpu()))
                target = output_dir / f'{name}_fake.jpg'
                img.save(target, format='JPEG', quality=args.jpeg_quality)
                n_written += 1
            if n_written % 100 == 0:
                print(f'  ... {n_written}/{len(ds)}')

    elapsed = time.time() - start
    print(f'[Infer] DONE: {n_written} images in {elapsed:.1f}s -> {output_dir}')


def collate_batch(batch, marker_idx):
    import torch as T
    return {
        'dapi': T.stack([b['dapi'] for b in batch]),
        'ihc': T.stack([b['ihc'] for b in batch]) if 'ihc' in batch[0] else T.zeros(len(batch), 3, 256, 256),
        'name': [b['name'] for b in batch],
        'marker_idx': T.full((len(batch),), marker_idx, dtype=T.long),
    }


if __name__ == '__main__':
    main()
