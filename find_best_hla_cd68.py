"""评估 HLA-DR 和 CD68 多个 epoch (skimage SSIM)"""
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


def evaluate(ckpt_path, marker, bs=8):
    model = build_pix2pix_model(3, 3, 3, 64).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()

    ds = DAPItoIHCDataset(root=data_root, marker=marker, split='val',
                          patch_size=256, augment=False)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)

    ssim_sum = 0.0
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
            n += fake.shape[0]

    del model
    torch.cuda.empty_cache()
    return ssim_sum / n


def scan(ckpt_dir_name, marker, epoch_list):
    print(f"\n=== {marker} ({ckpt_dir_name}) ===")
    results = []
    cd = ROOT / 'checkpoints' / ckpt_dir_name
    for name in epoch_list:
        if name == 'best':
            ck = cd / 'best.pt'
        else:
            ck = cd / f'epoch{name}.pt'
        if not ck.exists():
            print(f"  [SKIP] {ck.name} not found")
            continue
        s = evaluate(ck, marker)
        print(f"  {ck.name}: skimage SSIM = {s:.4f}")
        results.append((ck.name, s))
    return results


if __name__ == '__main__':
    # HLA-DR
    r1 = scan('pix2pix_v2_HLA-DR_1788962683', 'HLA-DR',
              ['best', 'epoch49', 'epoch54', 'epoch64', 'epoch74', 'epoch84', 'epoch94'])
    print('\nHLA-DR sorted:')
    for n, s in sorted(r1, key=lambda x: -x[1])[:3]:
        print(f'  {n}: {s:.4f}')

    # CD68
    r2 = scan('pix2pix_v2_CD68_1788969904', 'CD68',
              ['best', 'epoch34', 'epoch44', 'epoch54', 'epoch64', 'epoch74', 'epoch79'])
    print('\nCD68 sorted:')
    for n, s in sorted(r2, key=lambda x: -x[1])[:3]:
        print(f'  {n}: {s:.4f}')
