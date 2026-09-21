"""
使用训练好的 flow matching 模型对 test/DAPI 生成 IHC 染色并打包提交

Usage:
    python scripts/infer_fm_test.py \
        --ckpt checkpoints/HLA-DR_full_<id>/final.pt \
        --marker HLA-DR \
        --output-dir results/HLA-DR_fm_submission
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


from src.data.dataset import DAPItoIHCDataset
from src.models.flow_matching import FlowMatching, FlowMatchingConfig, build_model


MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']
SEMIFINAL_SEED = 2026
OFFICIAL_JPEG_QUALITY = 100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--marker', required=True, choices=MARKERS)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output-dir', default='results/HLA-DR_fm_submission')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-steps', type=int, default=50,
                   help='ODE sampling steps (overrides checkpoint default)')
    p.add_argument('--seed', type=int, choices=(SEMIFINAL_SEED,), default=SEMIFINAL_SEED)
    p.add_argument('--solver', choices=('euler', 'heun'), default='heun')
    return p.parse_args()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.batch_size < 1:
        raise ValueError('Invalid batch size')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[FM-Infer] marker={args.marker} ckpt={args.ckpt}')

    # 加载 checkpoint
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck['cfg']
    if cfg['defaults']['marker'] != args.marker:
        raise ValueError('Checkpoint marker does not match requested output marker')
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
          f'sampling steps={args.num_steps}, solver={args.solver}, seed={args.seed}')

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
    if not test_ds:
        raise ValueError('No test DAPI images')

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 输出目录
    out_dir = Path(args.output_dir) / 'results' / 'test' / args.marker
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Use an empty output directory to avoid mixing predictions')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[FM-Infer] output -> {out_dir}')

    # 推理循环
    t0 = time.time()
    saved = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            dapi = batch['dapi'].to(device)
            # Integrate the learned cleanward field; clamp only the final image.
            ihc = fm.sample(dapi, num_steps=args.num_steps, solver=args.solver)
            ihc = ihc.clamp(-1.0, 1.0)

            # 反归一化到 [0, 255] uint8
            ihc_np = ((ihc + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            ihc_np = np.transpose(ihc_np, (0, 2, 3, 1))  # NCHW -> NHWC

            # 取文件名（dataset 返回 dict 可能含 filename/path）
            # DAPItoIHCDataset 默认 __getitem__ 返回 {'dapi', 'ihc', 'name' 或 'path'}
            names = batch['name']

            for i in range(ihc_np.shape[0]):
                fname = names[i] if isinstance(names, (list, tuple)) else str(names)
                # 保证 .jpg 后缀
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fname = fname + '.jpg'
                out_path = out_dir / (Path(fname).stem + '_fake.jpg')
                Image.fromarray(ihc_np[i]).save(out_path, quality=OFFICIAL_JPEG_QUALITY,
                                                subsampling=0, optimize=False)
                saved += 1

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (batch_idx + 1) * (len(loader) - batch_idx - 1)
                print(f'[FM-Infer] batch {batch_idx + 1}/{len(loader)} saved={saved} '
                      f'elapsed={elapsed:.1f}s eta={eta:.1f}s')

    print(f'[FM-Infer] DONE. {saved} files saved in {out_dir} '
          f'(total time {time.time() - t0:.1f}s)')
    (Path(args.output_dir)/f'provenance_{args.marker}.json').write_text(json.dumps({
        'checkpoint': str(Path(args.ckpt).resolve()), 'marker': args.marker, 'images': saved,
        'num_steps': args.num_steps, 'solver': args.solver, 'seed': args.seed,
        'serialization': {'format': 'JPEG', 'quality': OFFICIAL_JPEG_QUALITY,
                          'subsampling': 0, 'optimize': False},
        'sampling_direction': 'positive cleanward velocity'
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
