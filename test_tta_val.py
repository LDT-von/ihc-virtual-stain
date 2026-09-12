"""快速验证 v10 TTA 是否真的提升 val SSIM"""
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

ROOT = Path(r'E:\aic\ihc-virtual-stain')
sys.path.insert(0, str(ROOT))

from src.data.dataset import DAPItoIHCDataset
from src.models.pix2pix_gan import build_pix2pix_model
from src.metrics.ssim_psnr import to_uint8
from skimage.metrics import structural_similarity as sk_ssim


def tta_flip(x, mode):
    if mode == 0: return x
    elif mode == 1: return torch.flip(x, [2])
    elif mode == 2: return torch.flip(x, [3])
    elif mode == 3: return torch.flip(x, [2, 3])
    return x

def tta_deflip(y, mode):
    if mode == 0: return y
    elif mode == 1: return torch.flip(y, [2])
    elif mode == 2: return torch.flip(y, [3])
    elif mode == 3: return torch.flip(y, [2, 3])
    return y


def eval_skim_no_tta(model, val_loader, device):
    model.eval()
    ssim_sum = 0.0
    n = 0
    with torch.inference_mode():
        for batch in val_loader:
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            fake = model.generator(dapi, dapi)
            for i in range(fake.shape[0]):
                p = to_uint8(fake[i].cpu())
                t = to_uint8(real[i].cpu())
                ssim_sum += sk_ssim(p, t, channel_axis=-1, data_range=255)
            n += fake.shape[0]
    return ssim_sum / n


def eval_skim_with_tta(models, val_loader, device, n_tta=4):
    for m in models: m.eval()
    ssim_sum = 0.0
    n = 0
    with torch.inference_mode():
        for batch in val_loader:
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            accum = None
            n_total = 0
            for model in models:
                for tta in range(n_tta):
                    x = tta_flip(dapi, tta)
                    fake = model.generator(x, x)
                    fake = tta_deflip(fake, tta)
                    if accum is None:
                        accum = fake
                    else:
                        accum = accum + fake
                    n_total += 1
            pred = accum / n_total
            for i in range(pred.shape[0]):
                p = to_uint8(pred[i].cpu())
                t = to_uint8(real[i].cpu())
                ssim_sum += sk_ssim(p, t, channel_axis=-1, data_range=255)
            n += pred.shape[0]
    return ssim_sum / n


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path('E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')

    # 测试每个 marker
    configs = {
        'CD45RO': ['checkpoints/pix2pix_v7_CD45RO_1789200177/epoch57.pt'],
        'Vimentin': ['checkpoints/pix2pix_v7_Vimentin_1789200192/epoch58.pt',
                     'checkpoints/pix2pix_v7_Vimentin_1789200192/epoch60.pt',
                     'checkpoints/pix2pix_v7_Vimentin_1789200192/epoch62.pt'],
        'HLA-DR': ['checkpoints/pix2pix_v2_HLA-DR_1788962683/best.pt'],
        'CD68': ['checkpoints/pix2pix_v2_CD68_1788969904/best.pt'],
    }

    for marker, ck_paths in configs.items():
        print(f'\n=== {marker} ===')
        models = []
        for ck_path in ck_paths:
            m = build_pix2pix_model(3, 3, 3, 64).to(device)
            ck = torch.load(ROOT / ck_path, map_location=device, weights_only=False)
            m.load_state_dict(ck['model'])
            models.append(m)

        val_ds = DAPItoIHCDataset(root=data_root, marker=marker,
                                   split='val', patch_size=256, augment=False)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

        # No TTA, single model
        t0 = time.time()
        ssim0 = eval_skim_no_tta(models[0], val_loader, device)
        print(f'  Single (no TTA): SSIM={ssim0:.4f} t={time.time()-t0:.0f}s')

        # 4 TTA, single model
        t0 = time.time()
        ssim1 = eval_skim_with_tta([models[0]], val_loader, device, n_tta=4)
        print(f'  Single + 4-flip TTA: SSIM={ssim1:.4f} t={time.time()-t0:.0f}s')

        # 4 TTA, all models ensemble
        if len(models) > 1:
            t0 = time.time()
            ssim2 = eval_skim_with_tta(models, val_loader, device, n_tta=4)
            print(f'  Ensemble {len(models)} + 4-flip TTA: SSIM={ssim2:.4f} t={time.time()-t0:.0f}s')

        for m in models:
            del m
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
