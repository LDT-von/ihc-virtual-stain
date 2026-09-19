"""
使用训练好的 flow matching 模型对 test/DAPI 生成 IHC 染色并打包提交

Usage:
    python scripts/infer_fm_test.py \
        --ckpt checkpoints/HLA-DR_full_<id>/final.pt \
        --marker HLA-DR \
        --output-dir results/HLA-DR_fm_submission
"""
import argparse
import builtins
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# 保存原始 print 后再替换，避免递归
_original_print = print
def _safe_print(*a, **k):
    k.setdefault('flush', True)
    try:
        _original_print(*a, **k)
    except UnicodeEncodeError:
        safe = [str(x).encode('gbk', errors='replace').decode('gbk') for x in a]
        _original_print(*safe, **k)


builtins.print = _safe_print

from src.data.dataset import DAPItoIHCDataset
from src.models.flow_matching import FlowMatching, FlowMatchingConfig, build_model


MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--marker', required=True, choices=MARKERS)
    p.add_argument('--data-root', default=r'E:\aic\初赛数据集（包含训练集和测试集输入）')
    p.add_argument('--output-dir', default='results/HLA-DR_fm_submission')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--num-steps', type=int, default=50,
                   help='ODE sampling steps; must match training cfg to be safe')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[FM-Infer] marker={args.marker} ckpt={args.ckpt}')

    # 加载 checkpoint
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck['cfg']
    model_cfg = cfg['model']
    fm_cfg = cfg.get('flow_matching', {})

    # 构建并加载模型
    model = build_model(
        in_channels=model_cfg['in_channels'],
        cond_channels=model_cfg['cond_channels'],
        base_channels=model_cfg['base_channels'],
        channel_mults=tuple(model_cfg['channel_mults']),
        num_res_blocks=model_cfg['num_res_blocks'],
        attention_resolutions=tuple(model_cfg['attention_resolutions']),
        dropout=model_cfg.get('dropout', 0.0),
    ).to(device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    print(f'[FM-Infer] loaded epoch {ck.get("epoch")}, '
          f'sampling steps={fm_cfg.get("num_sampling_steps", args.num_steps)}')

    # FlowMatching wrapper (for reverse ODE sampling)
    fm = FlowMatching(model, FlowMatchingConfig(
        sigma_min=fm_cfg.get('sigma_min', 1e-5),
        method=fm_cfg.get('method', 'optimal_transport'),
        num_sampling_steps=fm_cfg.get('num_sampling_steps', args.num_steps),
        solver=fm_cfg.get('solver', 'euler'),
    ))

    # 数据集 (test split, DAPI 输入)
    test_ds = DAPItoIHCDataset(
        root=Path(args.data_root),
        marker=args.marker,
        split='test',
        patch_size=cfg.get('data', {}).get('patch_size', 256),
        augment=False,
    )
    print(f'[FM-Infer] test set size: {len(test_ds)}')

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 输出目录
    out_dir = Path(args.output_dir) / args.marker
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[FM-Infer] output -> {out_dir}')

    # 推理循环
    t0 = time.time()
    saved = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            dapi = batch['dapi'].to(device)
            # fm.sample(condition=dapi) 返回 IHC RGB 图像, 已 clamp 到 [-1, 1]
            ihc = fm.sample(dapi)
            ihc = ihc.clamp(-1.0, 1.0)

            # 反归一化到 [0, 255] uint8
            ihc_np = ((ihc + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
            ihc_np = np.transpose(ihc_np, (0, 2, 3, 1))  # NCHW -> NHWC

            # 取文件名（dataset 返回 dict 可能含 filename/path）
            # DAPItoIHCDataset 默认 __getitem__ 返回 {'dapi', 'ihc', 'name' 或 'path'}
            names = batch.get('name') or batch.get('filename') or batch.get('path')
            if names is None:
                # 退路：用 batch index
                names = [f'unknown_{batch_idx * args.batch_size + i:04d}.jpg'
                         for i in range(ihc_np.shape[0])]

            for i in range(ihc_np.shape[0]):
                fname = names[i] if isinstance(names, (list, tuple)) else str(names)
                # 保证 .jpg 后缀
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fname = fname + '.jpg'
                out_path = out_dir / fname
                Image.fromarray(ihc_np[i]).save(out_path, quality=args.jpeg_quality)
                saved += 1

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (batch_idx + 1) * (len(loader) - batch_idx - 1)
                print(f'[FM-Infer] batch {batch_idx + 1}/{len(loader)} saved={saved} '
                      f'elapsed={elapsed:.1f}s eta={eta:.1f}s')

    print(f'[FM-Infer] DONE. {saved} files saved in {out_dir} '
          f'(total time {time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
