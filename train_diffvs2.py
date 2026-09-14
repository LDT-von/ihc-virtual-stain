"""
DiffVS 风格两阶段训练 — DAPI → 4 markers

Stage 1: NAFNet + Marker Embedding 联合训练（所有 marker 共享一个生成器）
Stage 2: 冻结 Stage1 编码器，单步 L1/L2 微调解码器

输入: DAPI (256x256)
输出: IHC (256x256)
条件: marker embedding (可学习, 4 个 marker)
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

import builtins
_oi = builtins.print
def _fp(*a, **k):
    k.setdefault('flush', True)
    try: _oi(*a, **k)
    except UnicodeEncodeError:
        safe = [str(x).encode('gbk', errors='replace').decode('gbk') for x in a]
        _oi(*safe, **k)
builtins.print = _fp

from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import to_uint8


# ============================================================================
# NAFNet Core
# ============================================================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x, g = x.chunk(2, dim=1)
        return x * g


class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class NAFBlock(nn.Module):
    """NAFNet block using Conv2d (matches official NAFNet architecture).
    
    FFN path: norm -> conv4(c→c*FFN_Expand) -> SimpleGate(→c*FFN_Expand/2) -> conv5(→c)
    SimpleGate halves channels → conv5 gets correct half-sized input.
    """
    def __init__(self, c, FFN_Expand=2.0):
        super().__init__()
        self.norm2 = LayerNorm2d(c)
        ffn_ch = int(c * FFN_Expand)
        self.conv4 = nn.Conv2d(c, ffn_ch, 1)
        self.conv5 = nn.Conv2d(ffn_ch // 2, c, 1)
        self.sg = SimpleGate()
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.conv5(self.sg(self.conv4(self.norm2(x))))
        return x + y * self.gamma


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True):
        super().__init__()
        if down:
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True))
        else:
            self.net = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class DiffVSNet(nn.Module):
    """
    Stage-1 模型: U-Net + NAFBlock + Marker 条件注入

    Marker 注入方式：AdaIN 风格 scale 调制（来自 DiffVS 的 MarkerTokenEncoder 简化版）
    """
    def __init__(self, in_ch=3, out_ch=3, width=64, num_markers=4, embed_dim=32):
        super().__init__()
        self.marker_embed = nn.Embedding(num_markers, embed_dim)

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, width, 3, 1, 1), nn.GELU())
        self.enc2 = nn.Sequential(nn.Conv2d(width, width*2, 4, 2, 1), nn.GELU())
        self.enc3 = nn.Sequential(nn.Conv2d(width*2, width*4, 4, 2, 1), nn.GELU(), NAFBlock(width*4), NAFBlock(width*4))
        self.enc4 = nn.Sequential(nn.Conv2d(width*4, width*8, 4, 2, 1), nn.GELU(), NAFBlock(width*8), NAFBlock(width*8), NAFBlock(width*8))

        # Decoder
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(width*8, width*4, 2, 2), nn.GELU(),
                                  NAFBlock(width*4), NAFBlock(width*4))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(width*8, width*2, 2, 2), nn.GELU(),
                                  NAFBlock(width*2), NAFBlock(width*2))
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(width*4, width, 2, 2), nn.GELU(),
                                  NAFBlock(width), NAFBlock(width))

        self.final = nn.Sequential(
            nn.Conv2d(width + embed_dim, width, 3, 1, 1), nn.GELU(),
            nn.Conv2d(width, out_ch, 3, 1, 1), nn.Tanh())

        self.embed_dim = embed_dim

    def _inject(self, x, marker_idx):
        emb = self.marker_embed(marker_idx)  # B, embed_dim
        # Match channel count: tile embed_dim → x.shape[1]
        n_ch = x.shape[1]
        scale = emb.view(-1, self.embed_dim, 1, 1)
        # Repeat to match channel dim if needed
        scale = scale.repeat(1, n_ch // self.embed_dim, 1, 1)
        return x + x * (scale * 0.2)

    def forward(self, x, marker_idx):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self._inject(self.dec3(e4), marker_idx)
        d3 = torch.cat([d3, e3], 1)
        d2 = self._inject(self.dec2(d3), marker_idx)
        d2 = torch.cat([d2, e2], 1)
        d1 = self._inject(self.dec1(d2), marker_idx)

        # Final: inject marker at output too
        d1 = torch.cat([d1, self.marker_embed.weight[marker_idx].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d1.shape[2], d1.shape[3])], 1)
        return self.final(d1)


# ============================================================================
# 数据集
# ============================================================================

class MultiMarkerDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, markers, split='train'):
        from src.data.dataset import DAPItoIHCDataset, load_image, build_transforms
        self.samples = []
        for mi, m in enumerate(markers):
            ds = DAPItoIHCDataset(data_root, m, split=split, patch_size=256, augment=(split == 'train'))
            transform = build_transforms(256, split == 'train')
            for dapi_p, ihc_p in ds.pairs:
                di = load_image(dapi_p)
                ii = load_image(ihc_p) if ihc_p else di
                t = transform(image=di, ihc=ii)
                self.samples.append({
                    'dapi': t['image'], 'ihc': t['ihc'], 'marker_idx': mi})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# 训练
# ============================================================================

def eval_model(model, loader, device, num_markers):
    model.eval()
    ssim_sum, n = 0.0, 0
    marker_ssim = {i: [] for i in range(num_markers)}
    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            midx = batch['marker_idx'].to(device)
            fake = model(dapi, midx)
            for i in range(fake.shape[0]):
                p = to_uint8(fake[i].cpu())
                t = to_uint8(real[i].cpu())
                ssim_sum += sk_ssim(p, t, channel_axis=-1, data_range=255)
                marker_ssim[midx[i].item()].append(ssim_sum)
                n += 1
    model.train()
    return {"ssim": ssim_sum / max(n, 1), "n": n,
            "per_marker": {i: sum(v)/len(v) if v else 0 for i, v in marker_ssim.items()}}


def eval_single_marker(model, marker, loader, device):
    model.eval()
    ssim_sum, n = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            midx = torch.full((dapi.size(0),), marker, dtype=torch.long, device=device)
            fake = model(dapi, midx)
            for i in range(fake.shape[0]):
                p = to_uint8(fake[i].cpu())
                t = to_uint8(real[i].cpu())
                ssim_sum += sk_ssim(p, t, channel_axis=-1, data_range=255)
                n += 1
    model.train()
    return ssim_sum / max(n, 1)


# Charbonnier loss (NAFNet 官方损失)
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps))


def collate_batch(batch, marker_idx=0):
    """Standalone collate function (must be top-level for Windows multiprocessing pickle)."""
    return {
        'dapi': torch.stack([b['dapi'] for b in batch]),
        'ihc':  torch.stack([b['ihc']  for b in batch]),
        'marker_idx': torch.tensor([b.get('marker_idx', marker_idx) for b in batch], dtype=torch.long)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--marker', type=str, default='HLA-DR')
    p.add_argument('--stage2', action='store_true')
    p.add_argument('--stage1-ckpt', type=str, default='')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--save-name', type=str, default='diffvs2')
    p.add_argument('--data-root', type=str,
                   default='E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']
    marker_idx = MARKERS.index(args.marker)
    data_root = Path(args.data_root)

    print('=' * 60)
    stage = 'Stage-2 (freeze encoder)' if args.stage2 else 'Stage-1'
    print(f'[{stage}] marker={args.marker} epochs={args.epochs} bs={args.batch_size} lr={args.lr}')
    print('=' * 60)

    # Data
    train_ds = DAPItoIHCDataset(data_root, args.marker, split='train', patch_size=256)
    val_ds   = DAPItoIHCDataset(data_root, args.marker, split='val',   patch_size=256)
    print(f'[Data] train={len(train_ds)}, val={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, drop_last=True, pin_memory=True,
                             collate_fn=lambda b: collate_batch(b, marker_idx))
    val_loader   = DataLoader(val_ds,   batch_size=8,  shuffle=False,
                             num_workers=0, pin_memory=True,
                             collate_fn=lambda b: collate_batch(b, marker_idx))

    # Model
    model = DiffVSNet(width=args.width, num_markers=len(MARKERS), embed_dim=32).to(device)
    print(f'[Model] params={sum(p.numel() for p in model.parameters()):,}')

    if args.stage2 and args.stage1_ckpt:
        print(f'[Stage-2] Loading stage-1 from {args.stage1_ckpt}')
        ck = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        # Freeze encoder
        for p in model.enc1.parameters(): p.requires_grad = False
        for p in model.enc2.parameters(): p.requires_grad = False
        for p in model.enc3.parameters(): p.requires_grad = False
        for p in model.enc4.parameters(): p.requires_grad = False
        for p in model.dec3.parameters(): p.requires_grad = False
        for p in model.dec2.parameters(): p.requires_grad = False
        trainable_params = list(model.dec1.parameters()) + list(model.final.parameters()) + list(model.marker_embed.parameters())
        print('[Stage-2] Frozen encoder, training dec1 + final + marker_embed')
    else:
        trainable_params = list(model.parameters())

    # Loss
    if args.stage2:
        loss_fn = CharbonnierLoss().to(device)  # Stage-2: Charbonnier (L2近似)
    else:
        loss_fn = nn.L1Loss().to(device)  # Stage-1: L1

    opt = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ts = int(time.time())
    out_dir = ROOT / 'checkpoints' / f'{args.save_name}_{args.marker}_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Init eval
    init_ssim = eval_single_marker(model, marker_idx, val_loader, device)
    print(f'[Init] val_ssim={init_ssim:.4f}')

    best_ssim = init_ssim
    best_epoch = -1

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        sum_g, n_b = 0.0, 0

        for bi, batch in enumerate(train_loader):
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            midx = batch['marker_idx'].to(device)

            opt.zero_grad(set_to_none=True)
            fake = model(dapi, midx)
            loss = loss_fn(fake, real)
            if not torch.isfinite(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            sum_g += loss.item()
            n_b += 1
            if (bi + 1) % 100 == 0:
                print(f'  [E{ep} B{bi+1}/{len(train_loader)}] G={loss.item():.4f} t={time.time()-t0:.0f}s')

        sched.step()
        ssim_v = eval_single_marker(model, marker_idx, val_loader, device)
        elapsed = time.time() - t0
        print(f'[Epoch {ep}] G={sum_g/max(n_b,1):.4f} val_ssim={ssim_v:.4f} t={elapsed:.0f}s')

        torch.save({'epoch': ep, 'model': model.state_dict(),
                    'val_ssim': ssim_v}, out_dir / f'epoch{ep}.pt')

        if ssim_v > best_ssim:
            best_ssim = ssim_v
            best_epoch = ep
            torch.save({'epoch': ep, 'model': model.state_dict(),
                        'val_ssim': ssim_v}, out_dir / 'best.pt')
            print(f'  [Best] epoch={ep} ssim={ssim_v:.4f}')

    print(f'\n[Done] Best val_ssim={best_ssim:.4f} (epoch {best_epoch})')


if __name__ == '__main__':
    main()
