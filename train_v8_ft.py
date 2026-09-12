"""v8 fine-tune：从已有最佳 checkpoint 继续训练少量 epochs

策略：
- 加载 CD45RO epoch57 (0.7031)
- 用小的 lr (1e-4)，稳定 loss
- 5-10 epochs，可能提升 +0.005~0.010
"""
import argparse
import sys
import time
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
from src.models.losses import PyramidL1SSIMLoss, GANLoss
from src.metrics.ssim_psnr import to_uint8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--marker', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--resume', type=str, required=True)
    p.add_argument('--lambda-pyr', type=float, default=200.0)
    p.add_argument('--lambda-ssim', type=float, default=80.0)
    p.add_argument('--d-every-n', type=int, default=4)
    p.add_argument('--save-prefix', default='pix2pix_v8_ft')
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
    print(f'[v8 Fine-tune] marker={args.marker} epochs={args.epochs} lr={args.lr}')
    print(f'[v8] resume={args.resume}')
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

    opt_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(model.discriminator.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))

    img_loss = PyramidL1SSIMLoss(pyramid_weight=args.lambda_pyr,
                                   ssim_weight=args.lambda_ssim, levels=4).to(device)
    gan_loss = GANLoss('lsgan').to(device)

    # Load resume - get initial SSIM
    ck = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])

    initial_ssim = eval_skim(model, val_loader, device)
    print(f'[Resume] SSIM={initial_ssim:.4f}')

    ts = int(time.time())
    out_dir = ROOT / 'checkpoints' / f'{args.save_prefix}_{args.marker}_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[Output] {out_dir}')

    # Best SSIM starts from initial
    best_ssim = initial_ssim
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sum_g, sum_d, n_b, n_d = 0.0, 0.0, 0, 0

        for bi, batch in enumerate(train_loader):
            dapi = batch['dapi'].to(device, non_blocking=True)
            real = batch['ihc'].to(device, non_blocking=True)

            if (bi + 1) % args.d_every_n == 0:
                opt_d.zero_grad(set_to_none=True)
                fake = model.generator(dapi, dapi)
                pr = model.discriminator(real, dapi)
                pf = model.discriminator(fake.detach(), dapi)
                ld = (gan_loss(pr, True) + gan_loss(pf, False)) * 0.5
                ld.backward()
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
                opt_d.step()
                sum_d += ld.item()
                n_d += 1

            opt_g.zero_grad(set_to_none=True)
            fake = model.generator(dapi, dapi)
            pf = model.discriminator(fake, dapi)
            lg = gan_loss(pf, True) * 0.5 + img_loss(fake, real)
            if not torch.isfinite(lg):
                continue
            lg.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            opt_g.step()

            sum_g += lg.item()
            n_b += 1

            if (bi + 1) % 100 == 0:
                print(f'  [E{epoch} B{bi+1}/{len(train_loader)}] G={lg.item():.3f} D={(sum_d/max(n_d,1)):.3f} t={time.time()-t0:.0f}s')

        avg_g = sum_g / max(n_b, 1)
        avg_d = sum_d / max(n_d, 1) if n_d > 0 else 0
        elapsed = time.time() - t0

        ssim_v = eval_skim(model, val_loader, device)
        print(f'[Epoch {epoch}] G={avg_g:.3f} D={avg_d:.3f} val_ssim={ssim_v:.4f} t={elapsed:.0f}s')

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
