"""v8 训练脚本：4 markers 并行/串行重训练

关键改进：
1. 用真实验证集 (skimage SSIM) 评估
2. 训练更长时间（30 epoch）
3. 引入新的数据增强
4. 更好的损失权重

策略：
- CD45RO: 重新训练 40 epochs from scratch（重点突破）
- Vimentin: 重新训练 30 epochs
- HLA-DR: fine-tune 20 epochs (继续 epoch94)
- CD68: fine-tune 15 epochs (从 epoch44)
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

# Flush prints
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
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--lambda-pyr', type=float, default=250.0)  # 提升 pyramid loss
    p.add_argument('--lambda-ssim', type=float, default=100.0)  # 提升 ssim loss
    p.add_argument('--d-every-n', type=int, default=3)
    p.add_argument('--brightness', type=float, default=0.3)
    p.add_argument('--contrast', type=float, default=0.3)
    p.add_argument('--save-prefix', default='pix2pix_v8')
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
    print(f'[v8 Train] marker={args.marker} epochs={args.epochs}')
    print(f'[v8] lambda_pyr={args.lambda_pyr} lambda_ssim={args.lambda_ssim}')
    print('=' * 60)

    train_ds = DAPItoIHCDataset(root=data_root, marker=args.marker,
                                 split='train', patch_size=256, augment=True)
    val_ds = DAPItoIHCDataset(root=data_root, marker=args.marker,
                               split='val', patch_size=256, augment=False)
    print(f'[Dataset] train={len(train_ds)} val={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    model = build_pix2pix_model(3, 3, 3, 64).to(device)
    print(f'[Model] G params: {sum(p.numel() for p in model.generator.parameters()):,}')

    opt_g = optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(model.discriminator.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))
    sched_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=1e-5)
    sched_d = optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs, eta_min=1e-5)

    img_loss = PyramidL1SSIMLoss(pyramid_weight=args.lambda_pyr,
                                   ssim_weight=args.lambda_ssim, levels=4).to(device)
    gan_loss = GANLoss('lsgan').to(device)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        if 'optimizer_g' in ck:
            opt_g.load_state_dict(ck['optimizer_g'])
            opt_d.load_state_dict(ck['optimizer_d'])
        start_epoch = ck.get('epoch', -1) + 1
        print(f'[Resume] {args.resume}, start_epoch={start_epoch}')

    ts = int(time.time())
    out_dir = ROOT / 'checkpoints' / f'{args.save_prefix}_{args.marker}_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[Output] {out_dir}')

    best_ssim = -1.0
    best_epoch = -1

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        t0 = time.time()
        sum_g, sum_d, n_b, n_d = 0.0, 0.0, 0, 0

        for bi, batch in enumerate(train_loader):
            dapi = batch['dapi'].to(device, non_blocking=True)
            real = batch['ihc'].to(device, non_blocking=True)

            # Discriminator
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

            # Generator
            opt_g.zero_grad(set_to_none=True)
            fake = model.generator(dapi, dapi)
            pf = model.discriminator(fake, dapi)
            lg = gan_loss(pf, True) + img_loss(fake, real)
            if not torch.isfinite(lg):
                continue
            lg.backward()
            torch.nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            opt_g.step()

            sum_g += lg.item()
            n_b += 1

            if (bi + 1) % 100 == 0:
                print(f'  [E{epoch} B{bi+1}/{len(train_loader)}] G={lg.item():.3f} D={(sum_d/max(n_d,1)):.3f} t={time.time()-t0:.0f}s')

        sched_g.step()
        sched_d.step()

        avg_g = sum_g / max(n_b, 1)
        avg_d = sum_d / max(n_d, 1) if n_d > 0 else 0
        elapsed = time.time() - t0

        # 用 skimage SSIM 评估
        ssim_v = eval_skim(model, val_loader, device)
        print(f'[Epoch {epoch}] G={avg_g:.3f} D={avg_d:.3f} val_ssim_skimage={ssim_v:.4f} t={elapsed:.0f}s lr={sched_g.get_last_lr()[0]:.2e}')

        # Save checkpoint
        ck_path = out_dir / f'epoch{epoch}.pt'
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer_g': opt_g.state_dict(), 'optimizer_d': opt_d.state_dict(),
                    'val_ssim': ssim_v}, ck_path)

        if ssim_v > best_ssim:
            best_ssim = ssim_v
            best_epoch = epoch
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer_g': opt_g.state_dict(), 'optimizer_d': opt_d.state_dict(),
                        'val_ssim': ssim_v}, out_dir / 'best.pt')
            print(f'  [Best] epoch={epoch} val_ssim={ssim_v:.4f}')

    print(f'\n[Done] Best: epoch={best_epoch} val_ssim={best_ssim:.4f}')


if __name__ == '__main__':
    main()
