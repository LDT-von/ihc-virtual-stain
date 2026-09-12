"""v9 改进策略：CD45RO 用纯图像损失微调（无 GAN）+ Mixup

理论依据：
- D loss=0 → 判别器已经太弱，GAN loss 反而干扰生成器
- 纯 L1 + SSIM + Pyramid 通常在病理图像上更稳定
- Mixup 混合两张图作为额外正则化
"""
import argparse
import sys
import time
import random
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as sk_ssim
import builtins

ROOT = Path(r'E:\aic\ihc-virtual-stain')
sys.path.insert(0, str(ROOT))

_orig_print = builtins.print
def _fp(*a, **k):
    k.setdefault('flush', True)
    try:
        _orig_print(*a, **k)
    except UnicodeEncodeError:
        safe = [str(x).encode('gbk', errors='replace').decode('gbk') for x in a]
        _orig_print(*safe, **k)
builtins.print = _fp

from src.data.dataset import DAPItoIHCDataset
from src.models.pix2pix_gan import build_pix2pix_model
from src.models.losses import PyramidL1SSIMLoss
from src.metrics.ssim_psnr import to_uint8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--marker', required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--resume', type=str, required=True)
    p.add_argument('--lambda-pyr', type=float, default=300.0)
    p.add_argument('--lambda-ssim', type=float, default=100.0)
    p.add_argument('--mixup-p', type=float, default=0.3)
    p.add_argument('--save-prefix', default='pix2pix_v9')
    p.add_argument('--data-root', default='E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
    return p.parse_args()


def eval_skim(model, val_loader, device):
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
    model.train()
    return ssim_sum / n


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)

    print('=' * 60)
    print(f'[v9 PureImg+Mixup] marker={args.marker} epochs={args.epochs} lr={args.lr}')
    print(f'[v9] resume={args.resume} mixup_p={args.mixup_p}')
    print('=' * 60)

    train_ds = DAPItoIHCDataset(root=data_root, marker=args.marker,
                                 split='train', patch_size=256, augment=True)
    val_ds = DAPItoIHCDataset(root=data_root, marker=args.marker,
                               split='val', patch_size=256, augment=False)
    print(f'[Dataset] train={len(train_ds)} val={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    model = build_pix2pix_model(3, 3, 3, 64).to(device)
    print(f'[Model] G params: {sum(p.numel() for p in model.generator.parameters()):,}')

    img_loss = PyramidL1SSIMLoss(pyramid_weight=args.lambda_pyr,
                                   ssim_weight=args.lambda_ssim, levels=4).to(device)

    # Load resume
    ck = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])

    initial_ssim = eval_skim(model, val_loader, device)
    print(f'[Resume] SSIM={initial_ssim:.4f}')

    opt_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)

    ts = int(time.time())
    out_dir = ROOT / 'checkpoints' / f'{args.save_prefix}_{args.marker}_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[Output] {out_dir}')

    best_ssim = initial_ssim
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sum_g, n_b = 0.0, 0

        for bi, batch in enumerate(train_loader):
            dapi = batch['dapi'].to(device, non_blocking=True)
            real = batch['ihc'].to(device, non_blocking=True)

            # Mixup: 30% 概率对 (input, target) 做线性混合
            if random.random() < args.mixup_p:
                lam = random.uniform(0.6, 0.9)
                perm = torch.randperm(dapi.size(0), device=device)
                dapi_mix = lam * dapi + (1 - lam) * dapi[perm]
                real_mix = lam * real + (1 - lam) * real[perm]
                opt_g.zero_grad(set_to_none=True)
                fake = model.generator(dapi_mix, dapi_mix)
                loss = img_loss(fake, real_mix)
            else:
                opt_g.zero_grad(set_to_none=True)
                fake = model.generator(dapi, dapi)
                loss = img_loss(fake, real)

            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            opt_g.step()

            sum_g += loss.item()
            n_b += 1

            if (bi + 1) % 200 == 0:
                print(f'  [E{epoch} B{bi+1}/{len(train_loader)}] G={loss.item():.3f} t={time.time()-t0:.0f}s')

        sched.step()
        avg_g = sum_g / max(n_b, 1)
        elapsed = time.time() - t0

        ssim_v = eval_skim(model, val_loader, device)
        print(f'[Epoch {epoch}] G={avg_g:.3f} val_ssim={ssim_v:.4f} t={elapsed:.0f}s')

        ck_path = out_dir / f'epoch{epoch}.pt'
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'val_ssim': ssim_v}, ck_path)

        if ssim_v > best_ssim:
            best_ssim = ssim_v
            best_epoch = epoch
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_ssim': ssim_v}, out_dir / 'best.pt')
            print(f'  [Best] epoch={epoch} val_ssim={ssim_v:.4f}')

    print(f'\n[Done] Initial {initial_ssim:.4f} -> Best {best_ssim:.4f} (epoch {best_epoch})')


if __name__ == '__main__':
    main()
