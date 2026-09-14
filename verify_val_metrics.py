"""用与训练 eval 完全一致的方式（skimage SSIM，data_range=255）验证。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim

from train_diffvs_latent_fm import DiffVSFull
from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import to_uint8

MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']
device = torch.device('cuda')

ckpts = {
    'HLA-DR': 'checkpoints/diffvs_hla_dr_HLA-DR_1789380785/best.pt',
    'CD68': 'checkpoints/diffvs_cd68_CD68_1789385241/best.pt',
    'CD45RO': 'checkpoints/diffvs_cd45ro_CD45RO_1789388694/best.pt',
    'Vimentin': 'checkpoints/diffvs_vim_Vimentin_1789392076/best.pt',
}

root = Path('E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')

model = DiffVSFull(base_ch=64, num_markers=4, marker_dim=32).to(device).eval()

for marker, ckpt_path in ckpts.items():
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'], strict=False)
    print(f'\n=== {marker} (saved_ssim={ck.get("ssim",-1):.4f} epoch={ck["epoch"]}) ===')

    ds = DAPItoIHCDataset(root, marker, split='val', patch_size=256, augment=False)
    midx = MARKERS.index(marker)

    ssim_sum, psnr_sum, n = 0.0, 0.0, 0
    with torch.inference_mode():
        for i in range(len(ds)):
            b = ds[i]
            dapi = b['dapi'].unsqueeze(0).to(device)
            real = b['ihc'].unsqueeze(0).to(device)
            midx_t = torch.full((1,), midx, dtype=torch.long, device=device)
            pred = model.forward_stage1(dapi, midx_t)
            p = to_uint8(pred[0].cpu())
            t = to_uint8(real[0].cpu())
            ssim_sum += sk_ssim(p, t, channel_axis=-1, data_range=255)
            mse = float(((p.astype(np.float32) - t.astype(np.float32)) ** 2).mean())
            psnr_sum += 10 * np.log10((255.0 ** 2) / max(mse, 1e-10))
            n += 1
            if (i+1) % 100 == 0:
                print(f'  ... {i+1}/{len(ds)}')

    print(f'  >>> val SSIM={ssim_sum/n:.4f} PSNR={psnr_sum/n:.2f} (n={n})')
