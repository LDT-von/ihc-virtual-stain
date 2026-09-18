"""
Stable Diffusion 虚拟染色模型推理脚本

功能:
- 从 DAPI 图像生成 IHC 图像
- 支持批量处理
- 支持多种采样方法 (DDIM/DDPM)
- 输出结果并打包成提交格式

Usage:
    python scripts/inference_sd.py --checkpoint outputs/sd_v1/best.pt --data-root /path/to/data --split test --output results/sd_v1
"""

import argparse
import builtins
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# 路径设置
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# 解决 Windows 编码问题
def _safe_print(*a, **k):
    k.setdefault('flush', True)
    try:
        print(*a, **k)
    except UnicodeEncodeError:
        safe = [str(x).encode('gbk', errors='replace').decode('gbk') for x in a]
        print(*safe, **k)
builtins.print = _safe_print

from src.models.stable_diffusion import VirtualStainDiffusion, StableDiffusionConfig
from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import to_uint8, ssim_gpu_single, psnr_gpu_single


# ============================================================================
# 数据加载
# ============================================================================

def build_dataloader(
    data_root: Path,
    marker: str,
    split: str,
    batch_size: int,
    patch_size: int = 256,
    num_workers: int = 4,
):
    """构建数据加载器"""
    dataset = DAPItoIHCDataset(
        root=data_root,
        marker=marker,
        split=split,
        patch_size=patch_size,
        augment=False,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader, len(dataset)


def collate_batch(batch):
    """批处理整理"""
    return {
        'dapi': torch.stack([b['dapi'] for b in batch]),
        'name': [b['name'] for b in batch],
    }


# ============================================================================
# 推理
# ============================================================================

def denormalize(x: torch.Tensor) -> np.ndarray:
    """
    将 [-1, 1] 范围的张量转换为 [0, 255] 的 uint8 图像
    """
    img = (x.clamp(-1, 1) + 1) * 127.5
    return img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


def save_images(
    images: np.ndarray,
    names: list,
    output_dir: Path,
):
    """
    保存图像
    
    Args:
        images: (B, H, W, C) uint8 数组
        names: 文件名列表
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img, name in zip(images, names):
        # 转为 PIL Image 并保存
        pil_img = Image.fromarray(img)
        pil_img.save(output_dir / f"{name}.png")


def inference(
    model: VirtualStainDiffusion,
    loader: DataLoader,
    device: torch.device,
    num_steps: int = 50,
    method: str = 'ddim',
    save_images_flag: bool = True,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    推理
    
    Returns:
        指标字典
    """
    model.eval()
    
    ssim_sum, psnr_sum, n = 0.0, 0.0, 0
    all_images = []
    all_names = []
    
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            dapi = batch['dapi'].to(device)
            names = batch['name']
            
            # 采样生成
            fake = model.sample(
                dapi,
                shape=(dapi.shape[0], 4, dapi.shape[2] // 8, dapi.shape[3] // 8),
                num_steps=num_steps,
                method=method,
            )
            
            # 转换并保存
            if save_images_flag and output_dir is not None:
                fake_np = denormalize(fake)
                save_images(fake_np, names, output_dir)
            
            all_images.append(fake.cpu())
            all_names.extend(names)
            
            n += dapi.shape[0]
    
    model.train()
    
    return {
        'n': n,
        'images': all_images,
        'names': all_names,
    }


def inference_with_metrics(
    model: VirtualStainDiffusion,
    loader: DataLoader,
    device: torch.device,
    num_steps: int = 50,
    method: str = 'ddim',
    save_images_flag: bool = True,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    推理并计算指标
    
    Returns:
        指标字典
    """
    model.eval()
    
    ssim_sum, psnr_sum, n = 0.0, 0.0, 0
    all_images = []
    all_names = []
    
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            names = batch['name']
            
            # 采样生成
            fake = model.sample(
                dapi,
                shape=(dapi.shape[0], 4, dapi.shape[2] // 8, dapi.shape[3] // 8),
                num_steps=num_steps,
                method=method,
            )
            
            # 计算指标
            for i in range(fake.shape[0]):
                ssim_val = ssim_gpu_single(fake[i], real[i])
                psnr_val = psnr_gpu_single(fake[i], real[i])
                ssim_sum += ssim_val
                psnr_sum += psnr_val
                n += 1
            
            # 转换并保存
            if save_images_flag and output_dir is not None:
                fake_np = denormalize(fake)
                save_images(fake_np, names, output_dir)
            
            all_images.append(fake.cpu())
            all_names.extend(names)
    
    model.train()
    
    return {
        'ssim': ssim_sum / n if n > 0 else 0.0,
        'psnr': psnr_sum / n if n > 0 else 0.0,
        'n': n,
        'images': all_images,
        'names': all_names,
    }


def pack_submission(output_dir: Path, zip_name: str = 'submission_sd.zip'):
    """
    打包提交文件
    """
    import zipfile
    
    results_dir = output_dir / 'test'
    if not results_dir.exists():
        print(f"[Warning] Results dir not found: {results_dir}")
        # 尝试在 output_dir 下查找
        results_dir = output_dir
    
    zip_path = output_dir.parent / zip_name
    
    print(f"[Pack] Creating {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img_path in sorted(results_dir.glob('*.png')):
            zf.write(img_path, img_path.name)
    
    print(f"[Pack] Created {zip_path} with {len(list(results_dir.glob('*.png')))} files")
    
    return zip_path


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stable Diffusion 虚拟染色推理')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型 checkpoint 路径')
    parser.add_argument('--config', type=str, default='',
                       help='配置 JSON 路径 (可选)')
    
    # 数据参数
    parser.add_argument('--data-root', type=str,
                       default='E:/aic/ihc-data',
                       help='数据根目录')
    parser.add_argument('--marker', type=str, default='HLA-DR',
                       help='IHC marker 名称')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='数据集划分')
    parser.add_argument('--patch-size', type=int, default=256,
                       help='Patch 尺寸')
    
    # 推理参数
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批大小')
    parser.add_argument('--num-steps', type=int, default=50,
                       help='采样步数')
    parser.add_argument('--method', type=str, default='ddim',
                       choices=['ddim', 'ddpm'],
                       help='采样方法')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载 worker 数')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='results/sd_v1',
                       help='输出目录')
    parser.add_argument('--no-save', action='store_true',
                       help='不保存图像')
    parser.add_argument('--pack-submission', action='store_true',
                       help='打包成提交文件')
    
    args = parser.parse_args()
    
    # 环境变量支持
    env_root = os.environ.get('IHC_DATA_ROOT')
    if env_root:
        args.data_root = env_root
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(f"[Inference] Stable Diffusion Virtual Staining")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Method: {args.method}")
    print(f"Num steps: {args.num_steps}")
    print("=" * 70)
    
    # 加载模型
    print("[Load] Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if 'config' in ckpt:
        config = ckpt['config']
        print(f"[Config] Loaded from checkpoint")
    else:
        # 默认配置
        config = StableDiffusionConfig()
        print(f"[Config] Using default config")
    
    model = VirtualStainDiffusion(config).to(device)
    model.load_state_dict(ckpt['model'])
    print(f"[Load] Model loaded, best SSIM: {ckpt.get('best_ssim', 'N/A')}")
    
    # 数据
    data_root = Path(args.data_root)
    loader, n_samples = build_dataloader(
        data_root, args.marker, args.split,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
    )
    
    print(f"Total samples: {n_samples}, Batches: {len(loader)}")
    
    # 输出目录
    output_dir = Path(args.output)
    if args.split == 'test':
        output_dir = output_dir / 'test'
    
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {output_dir}")
    
    # 推理计时
    start_time = time.time()
    
    if args.split in ['train', 'val']:
        # 有 ground truth，计算指标
        results = inference_with_metrics(
            model, loader, device,
            num_steps=args.num_steps,
            method=args.method,
            save_images_flag=not args.no_save,
            output_dir=output_dir,
        )
        
        print(f"\n[Results]")
        print(f"  Samples: {results['n']}")
        print(f"  SSIM: {results['ssim']:.4f}")
        print(f"  PSNR: {results['psnr']:.2f} dB")
    else:
        # 无 ground truth
        results = inference(
            model, loader, device,
            num_steps=args.num_steps,
            method=args.method,
            save_images_flag=not args.no_save,
            output_dir=output_dir,
        )
        
        print(f"\n[Results]")
        print(f"  Samples: {results['n']}")
    
    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.1f}s ({elapsed / max(results['n'], 1):.2f}s/sample)")
    
    # 打包提交
    if args.pack_submission and not args.no_save:
        zip_path = pack_submission(output_dir.parent, 'submission_sd.zip')
        print(f"\n[Submission] {zip_path}")
    
    print("\n[Done]")


if __name__ == '__main__':
    main()
