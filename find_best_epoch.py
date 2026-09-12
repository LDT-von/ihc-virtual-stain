"""评估某个 marker 的多个 epoch，找 skimage SSIM 最高的"""
import sys
import argparse
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


def evaluate_ckpt(ckpt_path, marker, batch_size=8):
    model = build_pix2pix_model(3, 3, 3, 64).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()

    ds = DAPItoIHCDataset(root=data_root, marker=marker, split='val',
                          patch_size=256, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir', required=True)
    p.add_argument('--marker', required=True)
    p.add_argument('--epochs', type=str, default='best,40,45,50,55')
    args = p.parse_args()

    ckpt_dir = ROOT / 'checkpoints' / args.ckpt_dir
    results = []
    target_epochs = [e.strip() for e in args.epochs.split(',')]

    for name in target_epochs:
        if name == 'best':
            ckpt = ckpt_dir / 'best.pt'
        else:
            ckpt = ckpt_dir / f'epoch{name}.pt'
        if not ckpt.exists():
            print(f'  [SKIP] {ckpt.name} not found')
            continue
        ssim = evaluate_ckpt(ckpt, args.marker)
        print(f'  {ckpt.name}: skimage SSIM = {ssim:.4f}')
        results.append((ckpt.name, ssim))

    print(f'\n=== Best checkpoint ===')
    results.sort(key=lambda x: -x[1])
    for name, s in results[:5]:
        print(f'  {name}: {s:.4f}')


if __name__ == '__main__':
    main()
