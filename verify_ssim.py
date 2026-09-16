"""验证两种 SSIM 实现的差异

跑 val 集，分别用：
- MetricAggregator (GPU 近似)
- skimage 标准 SSIM
- skimage SSIM with gaussian_weights=True
"""
import sys
from pathlib import Path

import torch

ROOT = Path(r'E:\aic\ihc-virtual-stain')
sys.path.insert(0, str(ROOT))
from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import MetricAggregator, ssim, to_uint8
from src.models.pix2pix_gan import build_pix2pix_model
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_root = Path(r'E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
ckpt = ROOT / 'checkpoints' / 'pix2pix_v2_HLA-DR_1788962683' / 'best.pt'

print('Loading model...')
model = build_pix2pix_model(3, 3, 3, 64).to(device)
ck = torch.load(ckpt, map_location=device, weights_only=False)
model.load_state_dict(ck['model'])
model.eval()

ds = DAPItoIHCDataset(root=data_root, marker='HLA-DR', split='val',
                      patch_size=256, augment=False)
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

from skimage.metrics import structural_similarity as sk_ssim
agg_gpu = MetricAggregator()
agg_skim_std = 0.0
agg_skim_gauss = 0.0
n = 0

with torch.inference_mode():
    for batch in loader:
        dapi = batch['dapi'].to(device)
        real = batch['ihc'].to(device)
        fake = model.generator(dapi, dapi)
        agg_gpu.update(fake, real)
        for i in range(fake.shape[0]):
            p = to_uint8(fake[i].cpu())
            t = to_uint8(real[i].cpu())
            agg_skim_std += sk_ssim(p, t, channel_axis=-1, data_range=255)
            agg_skim_gauss += sk_ssim(p, t, channel_axis=-1, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        n += fake.shape[0]

r = agg_gpu.result()
print(f'\nResults on {n} val samples:')
print(f'  GPU-approx SSIM (used during training): {r["ssim"]:.4f}')
print(f'  skimage std SSIM: {agg_skim_std/n:.4f}')
print(f'  skimage SSIM (gaussian_weights=True): {agg_skim_gauss/n:.4f}')
print(f'  PSNR (GPU): {r["psnr"]:.4f}')
