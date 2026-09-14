"""
NAFNet 基线训练脚本 — DAPI → IHC

NAFNet (ECCV 2022):
  - SimpleGate: 信道线性门控 (x * sigmoid(linear(x)))
  - NAFBlock: LayerNorm + channel-mixing FFN + SimpleGate
  - 无自注意力，无残差下采样，纯 CNN

来源：https://github.com/megvii-research/NAFNet
对比基线：与 DiffVS 的共享 marker 模型不同，这里每个 marker 独立训练。

输入: DAPI (3ch, 256x256)
输出: IHC (3ch, 256x256)
"""

import argparse
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np

import builtins
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
from src.metrics.ssim_psnr import to_uint8


# ============================================================================
# NAFNet 核心
# ============================================================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=1)
        return x * gate


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: B,C,H,W → B,H,W,C
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class NAFBlock(nn.Module):
    """NAFNet block using Conv2d (matches official NAFNet architecture).
    
    FFN path: norm -> conv4(c→c*FFN_Expand) -> SimpleGate(→c*FFN_Expand/2) -> conv5(→c)
    SimpleGate halves channels → conv5 gets correct half-sized input.
    """
    def __init__(self, c, FFN_Expand=2.0):
        super().__init__()
        self.norm2 = LayerNorm2d(c)
        ffn_ch = int(c * FFN_Expand)
        # FFN: c → ffn_ch → SimpleGate(halves to ffn_ch/2) → c
        self.conv4 = nn.Conv2d(c, ffn_ch, 1)          # c → c*FFN_Expand
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, 1)   # c*FFN_Expand/2 → c
        self.sg = SimpleGate()
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        # FFN branch: norm -> expand -> SimpleGate -> contract
        y = self.conv5(self.sg(self.conv4(self.norm2(x))))
        return x + y * self.gamma


class NAFNet(nn.Module):
    """NAFNet: encoder-decoder with NAFBlocks"""
    def __init__(self, in_ch=3, out_ch=3, width=64):
        super().__init__()
        # 编码器：下采样 3 次 → 8x (256→128→64→32)
        # width=64: enc1=64, enc2=128, enc3=256, enc4=512 (bottleneck)
        # FFN: c → c*4 → c (mid=4c)
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, width, 3, 1, 1), nn.GELU())  # 64
        self.enc2 = nn.Sequential(nn.Conv2d(width, width*2, 3, 2, 1), nn.GELU(),  # 128
                                   NAFBlock(width*2), NAFBlock(width*2))
        self.enc3 = nn.Sequential(nn.Conv2d(width*2, width*4, 3, 2, 1), nn.GELU(),  # 256
                                   NAFBlock(width*4), NAFBlock(width*4))
        self.enc4 = nn.Sequential(nn.Conv2d(width*4, width*8, 3, 2, 1), nn.GELU(),  # 512 (bottleneck)
                                   NAFBlock(width*8), NAFBlock(width*8), NAFBlock(width*8))

        # 解码器：上采样 3 次
        # enc3=256, enc4=512 → d3=256 → cat[e3]=256 → 512
        # d3=256, enc2=128 → d2=128 → cat[e2]=128 → 256
        # d2=128, enc1=64 → d1=64 → cat[e1]=64 → 128
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(width*8, width*4, 2, 2), nn.GELU(),  # 256
                                  NAFBlock(width*4), NAFBlock(width*4))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(width*8, width*2, 2, 2), nn.GELU(),  # 128
                                  NAFBlock(width*2), NAFBlock(width*2))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(width*4, width, 2, 2), nn.GELU(),   # 64
                                  NAFBlock(width), NAFBlock(width))

        self.final = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, 1, 1), nn.GELU(),
            nn.Conv2d(width, out_ch, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)    # B,64,256,256
        e2 = self.enc2(e1)   # B,128,128,128
        e3 = self.enc3(e2)   # B,256,64,64
        e4 = self.enc4(e3)   # B,512,32,32

        # Decoder with skip connections
        d3_raw = self.dec3(e4)   # B,256,64,64
        d3 = torch.cat([d3_raw, e3], 1)   # B,512,64,64

        d2_raw = self.dec2(d3)   # B,128,128,128
        d2 = torch.cat([d2_raw, e2], 1)   # B,256,128,128

        d1_raw = self.dec1(d2)   # B,64,256,256
        d1 = torch.cat([d1_raw, e1], 1)   # B,128,256,256

        return self.final(d1)


# ============================================================================
# 训练
# ============================================================================

class CombinedLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.05):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self._init_window()

    def _init_window(self):
        ks = 11
        sigma = 1.5
        gauss = torch.tensor([torch.exp(torch.tensor(-(x - ks//2)**2 / float(2*sigma**2))) for x in range(ks)])
        gauss /= gauss.sum()
        g2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        self.register_buffer('win', g2d.view(1, 1, ks, ks).expand(3, 1, ks, ks).contiguous())

    def _ssim(self, x, y):
        C1, C2 = 0.01**2, 0.03**2
        w = self.win.to(x.device)
        pad = 5
        mu_x = F.conv2d(x, w, padding=pad, groups=3)
        mu_y = F.conv2d(y, w, padding=pad, groups=3)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
        sig_x = F.conv2d(x**2, w, padding=pad, groups=3) - mu_x2
        sig_y = F.conv2d(y**2, w, padding=pad, groups=3) - mu_y2
        sig_xy = F.conv2d(x*y, w, padding=pad, groups=3) - mu_xy
        ssim_map = ((2*mu_xy+C1)*(2*sig_xy+C2)) / ((mu_x2+mu_y2+C1)*(sig_x+sig_y+C2))
        return ssim_map.mean()

    def forward(self, pred, target):
        loss = self.lambda_l1 * self.l1(pred, target)
        if self.lambda_ssim > 0:
            loss += self.lambda_ssim * (1 - self._ssim(pred, target))
        return loss


def eval_model(model, loader, device):
    model.eval()
    ssim_sum, psnr_sum, n = 0.0, 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            fake = model(dapi)
            for i in range(fake.shape[0]):
                p = to_uint8(fake[i].cpu())
                t = to_uint8(real[i].cpu())
                ssim_sum += sk_ssim(p, t, channel_axis=-1, data_range=255)
                mse = float(((p.astype(np.float32) - t.astype(np.float32))**2).mean())
                psnr_sum += 10 * np.log10((255.0**2) / max(mse, 1e-10))
                n += 1
    model.train()
    return ssim_sum / n, psnr_sum / n


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--marker', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--save-name', default='nafnet')
    p.add_argument('--data-root', default='E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)

    print('=' * 60)
    print(f'[NAFNet] marker={args.marker} epochs={args.epochs} bs={args.batch_size} lr={args.lr}')
    print(f'[NAFNet] device={device}')
    print('=' * 60)

    train_ds = DAPItoIHCDataset(data_root, args.marker, split='train', patch_size=256)
    val_ds   = DAPItoIHCDataset(data_root, args.marker, split='val',   patch_size=256)
    print(f'[Data] train={len(train_ds)}, val={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=8,  shuffle=False,
                              num_workers=0, pin_memory=True)

    model = NAFNet(in_ch=3, out_ch=3, width=args.width).to(device)
    print(f'[Model] params={sum(p.numel() for p in model.parameters()):,}')

    loss_fn = CombinedLoss(lambda_l1=1.0, lambda_ssim=0.05).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ts = int(time.time())
    out_dir = ROOT / 'checkpoints' / f'{args.save_name}_{args.marker}_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ssim = -1.0
    init_ssim, init_psnr = eval_model(model, val_loader, device)
    print(f'[Init] val_ssim={init_ssim:.4f}, val_psnr={init_psnr:.2f}')

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sum_g, n_b = 0.0, 0

        for bi, batch in enumerate(train_loader):
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)

            opt.zero_grad(set_to_none=True)
            fake = model(dapi)
            loss = loss_fn(fake, real)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            sum_g += loss.item()
            n_b += 1
            if (bi + 1) % 100 == 0:
                print(f'  [E{epoch} B{bi+1}/{len(train_loader)}] G={loss.item():.4f} t={time.time()-t0:.0f}s')

        sched.step()
        elapsed = time.time() - t0
        avg_g = sum_g / max(n_b, 1)

        ssim_v, psnr_v = eval_model(model, val_loader, device)
        print(f'[Epoch {epoch}] G={avg_g:.4f} val_ssim={ssim_v:.4f} val_psnr={psnr_v:.2f} t={elapsed:.0f}s')

        ck = {'epoch': epoch, 'model': model.state_dict(),
              'val_ssim': ssim_v, 'val_psnr': psnr_v}
        torch.save(ck, out_dir / f'epoch{epoch}.pt')

        if ssim_v > best_ssim:
            best_ssim = ssim_v
            torch.save(ck, out_dir / 'best.pt')
            print(f'  [Best] epoch={epoch} ssim={ssim_v:.4f} psnr={psnr_v:.2f}')

    print(f'\n[Done] Best val_ssim={best_ssim:.4f}')


if __name__ == '__main__':
    main()
