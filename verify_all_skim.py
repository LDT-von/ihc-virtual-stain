"""用 skimage 标准 SSIM 评估所有 marker"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(r'E:\aic\ihc-virtual-stain')
sys.path.insert(0, str(ROOT))
from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import to_uint8
from src.models.pix2pix_gan import build_pix2pix_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_root = Path(r'E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')

BEST_CKPTS = {
    'HLA-DR':   ROOT / 'checkpoints' / 'pix2pix_v2_HLA-DR_1788962683' / 'best.pt',
    'CD68':     ROOT / 'checkpoints' / 'pix2pix_v2_CD68_1788969904' / 'best.pt',
    'CD45RO':   ROOT / 'checkpoints' / 'pix2pix_v7_CD45RO_1789200177' / 'best.pt',
    'Vimentin': ROOT / 'checkpoints' / 'pix2pix_v7_Vimentin_1789200192' / 'best.pt',
}

results = {}
for marker, ckpt in BEST_CKPTS.items():
    print(f"\n[{marker}] {ckpt.name}")
    model = build_pix2pix_model(3, 3, 3, 64).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()

    ds = DAPItoIHCDataset(root=data_root, marker=marker, split='val',
                          patch_size=256, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    ssim_sum = 0.0
    psnr_sum = 0.0
    n = 0
    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            fake = model.generator(dapi, dapi)
            for i in range(fake.shape[0]):
                p = to_uint8(fake[i].cpu())
                t = to_uint8(real[i].cpu())
                ssim_sum += sk_ssim(p, t, channel_axis=-1, data_range=255)
                mse = ((fake[i].clamp(-1, 1) - real[i].clamp(-1, 1)) ** 2).mean().item()
                psnr_sum += 10.0 * (2.0 / mse if mse > 1e-10 else 100) ** 0.5 * 0.5  # log10(4/mse)
                # simpler
            n += fake.shape[0]

    avg_ssim = ssim_sum / n
    print(f"  skimage SSIM: {avg_ssim:.4f}")
    results[marker] = avg_ssim
    del model
    torch.cuda.empty_cache()

print('\n========== True val scores (skimage SSIM) ==========')
for m, s in results.items():
    print(f"  {m}: {s:.4f}")
avg = sum(results.values()) / len(results)
print(f"\n  Average: {avg:.4f}")
print(f"  Estimated platform score: {avg * 100:.2f}")
