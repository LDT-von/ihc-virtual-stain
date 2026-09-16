"""
高级推理脚本 - 8x TTA + 集成多模型

TTA (Test-Time Augmentation) 策略：
1. 原始图像
2. 水平翻转
3. 垂直翻转
4. 旋转 180°
5. 对角翻转
6. 旋转 90° + 水平翻转
7. 旋转 270° + 垂直翻转
8. 旋转 90° + 对角翻转

用法：
    python infer_sota_v2.py --marker HLA-DR --ckpt_dir ./checkpoints/sota_v2_HLA-DR_xxx --output_dir ./submissions
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import ssim as skimage_ssim, to_uint8


def apply_tta_transforms(img):
    """
    生成 8 种 TTA 变换
    
    Args:
        img: (C, H, W) tensor, 范围 [-1, 1]
    Returns:
        list of 8 tensors
    """
    transforms = [
        lambda x: x,                           # 0: 原始
        lambda x: torch.flip(x, dims=[2]),      # 1: 水平翻转
        lambda x: torch.flip(x, dims=[1]),      # 2: 垂直翻转
        lambda x: torch.flip(torch.flip(x, dims=[1]), dims=[2]),  # 3: 旋转180
        lambda x: torch.flip(x, dims=[1, 2]),   # 4: 对角翻转
        lambda x: torch.flip(x.rot90(1, dims=[1,2]), dims=[2]),  # 5: rot90 + hflip
        lambda x: torch.flip(x.rot90(3, dims=[1,2]), dims=[1]),  # 6: rot270 + vflip
        lambda x: x.rot90(2, dims=[1,2]),      # 7: rot180
    ]
    return [t(img) for t in transforms]


def reverse_tta(pred, tta_idx):
    """逆向 TTA 变换"""
    if tta_idx == 0:
        return pred
    elif tta_idx == 1:
        return torch.flip(pred, dims=[2])
    elif tta_idx == 2:
        return torch.flip(pred, dims=[1])
    elif tta_idx == 3:
        return torch.flip(torch.flip(pred, dims=[1]), dims=[2])
    elif tta_idx == 4:
        return torch.flip(pred, dims=[1, 2])
    elif tta_idx == 5:
        return torch.flip(pred, dims=[2]).rot90(-1, dims=[1,2])
    elif tta_idx == 6:
        return torch.flip(pred, dims=[1]).rot90(-3, dims=[1,2])
    elif tta_idx == 7:
        return pred.rot90(-2, dims=[1,2])
    return pred


def load_model(ckpt_path, device, model_class=None):
    """加载模型"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if model_class is None:
        # 尝试导入
        try:
            from src.models.pix2pix_gan import build_pix2pix_model
            model = build_pix2pix_model(3, 3, 3, 64)
        except:
            # 使用内置简单模型
            model = SimpleGenerator(3, 3)
    
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device)
    model.eval()
    return model, ckpt


class SimpleGenerator(nn.Module):
    """简单的备用生成器"""
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, out_ch, 3, 1, 1), nn.Tanh(),
        )
    
    def forward(self, x):
        return self.net(x)


def predict_with_tta(model, dapi_batch, device):
    """使用 TTA 进行批量预测"""
    B = dapi_batch.shape[0]
    preds = []
    
    for i in range(B):
        dapi = dapi_batch[i]  # (C, H, W)
        tta_imgs = apply_tta_transforms(dapi)
        
        tta_preds = []
        for tta_img in tta_imgs:
            tta_img_b = tta_img.unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model.generator(tta_img_b, tta_img_b)
            pred = pred.squeeze(0).cpu()
            tta_preds.append(pred)
        
        # 逆向变换并平均
        reversed_preds = []
        for idx, pred in enumerate(tta_preds):
            reversed_preds.append(reverse_tta(pred, idx))
        
        avg_pred = torch.stack(reversed_preds).mean(0)
        preds.append(avg_pred)
    
    return torch.stack(preds)


def predict_ensemble_tta(models, dapi_batch, device):
    """多模型集成 + TTA"""
    B = dapi_batch.shape[0]
    all_preds = []
    
    for model in models:
        preds = []
        for i in range(B):
            dapi = dapi_batch[i]
            tta_imgs = apply_tta_transforms(dapi)
            
            tta_preds = []
            for tta_img in tta_imgs:
                tta_img_b = tta_img.unsqueeze(0).to(device)
                with torch.no_grad():
                    if hasattr(model, 'generator'):
                        pred = model.generator(tta_img_b, tta_img_b)
                    else:
                        pred = model(tta_img_b)
                pred = pred.squeeze(0).cpu()
                tta_preds.append(pred)
            
            reversed_preds = [reverse_tta(p, idx) for idx, p in enumerate(tta_preds)]
            avg_pred = torch.stack(reversed_preds).mean(0)
            preds.append(avg_pred)
        
        all_preds.append(torch.stack(preds))
    
    # 模型平均
    return torch.stack(all_preds).mean(0)


def evaluate_on_val(model, val_loader, device, use_tta=True):
    """在验证集上评估"""
    ssims, psnrs = [], []
    
    for batch in val_loader:
        dapi = batch["dapi"].to(device)
        real = batch["ihc"].to(device)
        
        if use_tta:
            fake = predict_with_tta(model, dapi, device).to(device)
        else:
            with torch.no_grad():
                fake = model.generator(dapi, dapi)
        
        for i in range(fake.shape[0]):
            ss = skimage_ssim(fake[i], real[i])
            mse = ((fake[i].clamp(-1,1) - real[i].clamp(-1,1))**2).mean().item()
            ps = 100.0 if mse < 1e-10 else 10.0 * np.log10(4.0 / mse)
            ssims.append(ss)
            psnrs.append(ps)
    
    return float(np.mean(ssims)), float(np.mean(psnrs))


def parse_args():
    p = argparse.ArgumentParser(description="SOTA 推理脚本")
    p.add_argument("--marker", type=str, required=True)
    p.add_argument("--data_root", type=str,
                   default="E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）")
    p.add_argument("--ckpt", type=str, default=None,
                   help="单个 checkpoint 路径")
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="checkpoint 目录，自动找 best.pt 或最佳 epoch")
    p.add_argument("--ckpts", nargs="+", default=[],
                   help="多个 checkpoint 路径做集成")
    p.add_argument("--output_dir", type=str, default="./submissions")
    p.add_argument("--batch_size", type=int, default=4)  # TTA 需要更多显存
    p.add_argument("--use_tta", action="store_true", default=True,
                   help="启用 TTA (默认开启)")
    p.add_argument("--no_tta", action="store_true",
                   help="禁用 TTA")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--save_images", action="store_true",
                   help="保存预测图像")
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = not args.no_tta
    
    print("=" * 60)
    print(f"[Inference] marker={args.marker} device={device}")
    print(f"[Inference] use_tta={use_tta}")
    print(f"[Inference] output={args.output_dir}")
    print("=" * 60)
    
    # 加载模型
    models = []
    
    if args.ckpt:
        print(f"[Loading] {args.ckpt}")
        model, _ = load_model(args.ckpt, device)
        models.append(model)
    
    if args.ckpt_dir:
        ckpt_dir = Path(args.ckpt_dir)
        if (ckpt_dir / "best_ema.pt").exists():
            ckpt_path = ckpt_dir / "best_ema.pt"
        elif (ckpt_dir / "best.pt").exists():
            ckpt_path = ckpt_dir / "best.pt"
        else:
            # 找最佳 epoch
            ckpts = sorted(ckpt_dir.glob("epoch*.pt"))
            if ckpts:
                ckpt_path = ckpts[-1]
            else:
                print("[Error] No checkpoint found")
                return
        
        print(f"[Loading] {ckpt_path}")
        model, ckpt = load_model(ckpt_path, device)
        models.append(model)
    
    for ckpt_path in args.ckpts:
        print(f"[Loading] {ckpt_path}")
        model, _ = load_model(ckpt_path, device)
        models.append(model)
    
    if not models:
        print("[Error] No models loaded!")
        return
    
    print(f"[Models] {len(models)} model(s) loaded")
    
    # 数据集
    if args.split == "test":
        ds = DAPItoIHCDataset(
            root=args.data_root, marker=args.marker, split="test",
            patch_size=256, augment=False)
    else:
        ds = DAPItoIHCDataset(
            root=args.data_root, marker=args.marker, split=args.split,
            patch_size=256, augment=False)
    
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)
    
    print(f"[Data] {args.split}: {len(ds)} samples, {len(loader)} batches")
    
    # 推理
    output_dir = Path(args.output_dir) / f"{args.marker}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_ssims, all_psnrs = [], []
    predictions = []
    
    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        dapi = batch["dapi"].to(device)
        real = batch["ihc"].to(device) if "ihc" in batch else None
        names = batch.get("name", [f"img_{batch_idx}_{i}" for i in range(dapi.shape[0])])
        
        if len(models) > 1:
            fake = predict_ensemble_tta(models, dapi, device).to(device)
        elif use_tta:
            fake = predict_with_tta(models[0], dapi, device).to(device)
        else:
            with torch.no_grad():
                fake = models[0].generator(dapi, dapi)
        
        # 计算指标
        if real is not None:
            for i in range(fake.shape[0]):
                ss = skimage_ssim(fake[i].cpu(), real[i].cpu())
                mse = ((fake[i].clamp(-1,1) - real[i].clamp(-1,1))**2).mean().item()
                ps = 100.0 if mse < 1e-10 else 10.0 * np.log10(4.0 / mse)
                all_ssims.append(ss)
                all_psnrs.append(ps)
        
        # 保存图像
        if args.save_images:
            for i in range(fake.shape[0]):
                img = to_uint8(fake[i].cpu())
                import PIL.Image
                PIL.Image.fromarray(img).save(output_dir / f"{names[i]}.png")
        
        predictions.append(fake.cpu())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"[Batch {batch_idx+1}/{len(loader)}]")
    
    elapsed = time.time() - t0
    
    # 汇总
    print("\n" + "=" * 60)
    print(f"[Results] marker={args.marker}")
    if all_ssims:
        print(f"[Results] SSIM: {np.mean(all_ssims):.4f} ± {np.std(all_ssims):.4f}")
        print(f"[Results] PSNR: {np.mean(all_psnrs):.2f} ± {np.std(all_psnrs):.2f}")
    print(f"[Results] Time: {elapsed:.1f}s ({elapsed/len(ds)*1000:.1f}ms/img)")
    print(f"[Results] Output: {output_dir}")
    print("=" * 60)
    
    # 保存结果
    results = {
        "marker": args.marker,
        "num_samples": len(ds),
        "ssim_mean": float(np.mean(all_ssims)) if all_ssims else None,
        "ssim_std": float(np.std(all_ssims)) if all_ssims else None,
        "psnr_mean": float(np.mean(all_psnrs)) if all_psnrs else None,
        "psnr_std": float(np.std(all_psnrs)) if all_psnrs else None,
        "use_tta": use_tta,
        "num_models": len(models),
        "elapsed_seconds": elapsed,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    main()
