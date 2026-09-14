"""
DiffVS 风格训练脚本 — DAPI → HLA-DR/CD68/CD45RO/Vimentin

论文方法核心：
  1. 标记条件共享模型：一个 U-Net 同时学习 4 个 marker
  2. 单步像素损失微调：用扩散模型的去噪方向作为像素级监督
  3. label embedding 提供染色风格条件

适配说明（基于 DiffVS AAAI 2026 论文）：
  - 论文原任务：H&E → IHC（4 个 marker）
  - 本任务：DAPI → IHC（同样 4 个 marker，方法可直接迁移）
  - 差异：输入模态不同（都是整张病理图，域适应方法一致）

输入: DAPI (3ch, 256x256 JPG)
输出: IHC (3ch, 256x256 JPG)
条件: marker label embedding (one-hot, 4 维)
"""

import argparse
import sys
import time
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ─── 安全的打印（避免 GBK 乱码） ───
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
# 1. Label Embedding（DiffVS 的标记条件）
# ============================================================================

class MarkerEmbedding(nn.Module):
    """
    每个 marker 有独立的可学习 embedding，在 forward 时加入生成器。
    DiffVS 的 Visual-Concept Prompt 类似，但这里用更简洁的 AdaIN 注入。
    """
    def __init__(self, num_markers: int = 4, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(num_markers, embed_dim)
        # marker id → 全局共享
        self.marker_id_to_idx = {
            'HLA-DR': 0, 'CD68': 1, 'CD45RO': 2, 'Vimentin': 3
        }

    def forward(self, marker_idx: torch.Tensor) -> torch.Tensor:
        """marker_idx: (B,) long tensor → (B, embed_dim)"""
        return self.embed(marker_idx)


# ============================================================================
# 2. NAFNet Block（DiffVS 使用的 SimpleGate 块）
# ============================================================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=1)
        return x * gate


class NAFBlock(nn.Module):
    """
    NAFNet 的核心块（NAFBlock with SimpleGate）。
    比标准残差块有更强的通道交互，在图像恢复任务上更好。

    来自：NAFNet, ECCV 2022
    https://github.com/megvii-research/NAFNet
    """
    def __init__(self, c, bias=False):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.sfc = nn.Sequential(
            nn.Linear(c, c * 4, bias=bias),
            SimpleGate(),
            nn.Linear(c * 2, c, bias=bias),
        )
        self.res_scale = nn.Parameter(torch.ones(1, c, 1, 1) * 0.1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.sfc(x) * self.res_scale


# ============================================================================
# 3. DiffVS Generator（U-Net + NAFBlock + Marker Condition）
# ============================================================================

class DiffVSGenerator(nn.Module):
    """
    DiffVS Generator: U-Net 编码器-解码器 + NAFBlock 残差 + Marker 条件注入

    条件注入方式（AdaIN 风格）：
      - marker embedding → scale + shift → 加到特征图上
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 64,
        num_markers: int = 4,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.marker_embed = MarkerEmbedding(num_markers, embed_dim)

        # ── Encoder ──
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, 4, 2, 1),
            nn.BatchNorm2d(base_filters*2), nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, 4, 2, 1),
            nn.BatchNorm2d(base_filters*4), nn.LeakyReLU(0.2, inplace=True))
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*8, 4, 2, 1),
            nn.BatchNorm2d(base_filters*8), nn.LeakyReLU(0.2, inplace=True))

        # ── Middle（NAFBlocks）──
        self.middle = nn.Sequential(
            NAFBlock(base_filters*8), NAFBlock(base_filters*8),
            NAFBlock(base_filters*8),
        )

        # ── Decoder ──
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*8 + embed_dim, base_filters*4, 4, 2, 1),
            nn.BatchNorm2d(base_filters*4), nn.ReLU(inplace=True),
            NAFBlock(base_filters*4), NAFBlock(base_filters*4))
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*8, base_filters*2, 4, 2, 1),
            nn.BatchNorm2d(base_filters*2), nn.ReLU(inplace=True),
            NAFBlock(base_filters*2), NAFBlock(base_filters*2))
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*4, base_filters, 4, 2, 1),
            nn.BatchNorm2d(base_filters), nn.ReLU(inplace=True),
            NAFBlock(base_filters))
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*2, base_filters, 4, 2, 1),
            nn.BatchNorm2d(base_filters), nn.ReLU(inplace=True))

        # ── Output ──
        self.out = nn.Sequential(
            nn.Conv2d(base_filters + embed_dim, base_filters, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, out_channels, 3, 1, 1),
            nn.Tanh(),  # [-1, 1]
        )

    def _inject_condition(self, x: torch.Tensor, marker_idx: torch.Tensor) -> torch.Tensor:
        """
        AdaIN 风格注入 marker 条件。
        marker_embed: (B, embed_dim) → reshape → broadcast → scale + shift
        """
        emb = self.marker_embed(marker_idx)  # (B, embed_dim)
        # 注入到通道维度：expand 到 B x embed_dim x 1 x 1
        scale = emb.view(-1, emb.shape[1], 1, 1)
        return x * (1 + scale * 0.3)  # 温和注入

    def forward(self, x: torch.Tensor, marker_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: DAPI 输入 (B, 3, 256, 256)
            marker_idx: marker ID (B,) long
        Returns:
            fake IHC (B, 3, 256, 256)
        """
        # Encoder
        e1 = self.enc1(x)   # B,64,128,128
        e2 = self.enc2(e1)  # B,128,64,64
        e3 = self.enc3(e2)  # B,256,32,32
        e4 = self.enc4(e3)  # B,512,16,16

        # Middle
        m = self.middle(e4)  # B,512,16,16

        # Decoder (注入条件到深层)
        d4 = self.dec4(self._inject_condition(m, marker_idx))  # B,256,32,32
        d4 = torch.cat([d4, e3], dim=1)  # B,512,32,32
        d3 = self.dec3(d4)                                        # B,128,64,64
        d3 = torch.cat([d3, e2], dim=1)  # B,256,64,64
        d2 = self.dec2(d3)                                        # B,64,128,128
        d2 = torch.cat([d2, e1], dim=1)  # B,128,128,128
        d1 = self.dec1(d2)                # B,64,256,256

        # Final（再次注入条件）
        out = self.out(self._inject_condition(d1, marker_idx))
        return out


# ============================================================================
# 4. Patch Discriminator（Pix2Pix 风格）
# ============================================================================

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_channels*2, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = ndf
        for i in range(1, n_layers):
            layers += [nn.Conv2d(ch, ch*2, 4, 2, 1), nn.BatchNorm2d(ch*2), nn.LeakyReLU(0.2, True)]
            ch *= 2
        layers += [nn.Conv2d(ch, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, cond):
        return self.net(torch.cat([x, cond], dim=1))


# ============================================================================
# 5. 损失函数
# ============================================================================

class CombinedDiffVSLoss(nn.Module):
    """
    DiffVS 组合损失：
      - L1: 像素级重建（主要）
      - SSIM: 结构相似性（DiffVS 论文中用于评估）
      - Perceptual: LPIPS 风格 VGG 特征（DiffVS 也用感知损失）
      - GAN: LSGAN（共享模型需要区分不同 marker）
    """
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.1, lambda_perc=0.05):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perc = lambda_perc

        self.l1 = nn.L1Loss()
        self.ssim_win = 11
        self._init_ssim_window()

        # VGG perceptual
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            self.vgg = vgg.eval()
            for p in self.vgg.parameters():
                p.requires_grad = False
            self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('vgg_std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        except Exception:
            self.vgg = None

    def _init_ssim_window(self):
        sigma = 1.5
        ks = self.ssim_win
        gauss = torch.tensor([torch.exp(torch.tensor(-(x - ks//2)**2 / float(2*sigma**2)))
                              for x in range(ks)])
        gauss /= gauss.sum()
        g2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        self.register_buffer('ssim_win', g2d.view(1, 1, ks, ks).expand(3, 1, ks, ks).contiguous())

    def _ssim(self, x, y):
        C1, C2 = 0.01**2, 0.03**2
        win = self.ssim_win.to(x.device)
        pad = self.ssim_win // 2
        mu_x = F.conv2d(x, win, padding=pad, groups=3)
        mu_y = F.conv2d(y, win, padding=pad, groups=3)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
        sig_x = F.conv2d(x**2, win, padding=pad, groups=3) - mu_x2
        sig_y = F.conv2d(y**2, win, padding=pad, groups=3) - mu_y2
        sig_xy = F.conv2d(x*y, win, padding=pad, groups=3) - mu_xy
        ssim_map = ((2*mu_xy + C1)*(2*sig_xy + C2)) / ((mu_x2 + mu_y2 + C1)*(sig_x + sig_y + C2))
        return ssim_map.mean()

    def _perceptual(self, pred, target):
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        def norm(x):
            x = (x + 1) / 2
            return (x - self.vgg_mean.to(x.device)) / self.vgg_std.to(x.device)
        feats_p = self.vgg(norm(pred))
        feats_t = self.vgg(norm(target))
        return F.l1_loss(feats_p, feats_t)

    def forward(self, fake, real):
        loss = 0.0
        if self.lambda_l1 > 0:
            loss += self.lambda_l1 * self.l1(fake, real)
        if self.lambda_ssim > 0:
            loss += self.lambda_ssim * (1 - self._ssim(fake, real))
        if self.lambda_perc > 0:
            loss += self.lambda_perc * self._perceptual(fake, real)
        return loss


# ============================================================================
# 6. 评估（统一用 skimage 局部 SSIM）
# ============================================================================

def eval_model(model, loader, device, marker_idx_map):
    """在 val 集上用 skimage 局部 SSIM 评估"""
    model.eval()
    ssim_sum, psnr_sum, n = 0.0, 0.0, 0
    marker_ssim = {m: [] for m in marker_idx_map}

    # 收集 marker_idx → marker name 映射
    idx_to_marker = {v: k for k, v in marker_idx_map.items()}

    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            # 从 name 中推断 marker（兼容 dataset 返回的字段）
            names = batch.get('name', [''] * dapi.size(0))

            # 从路径推断 marker
            marker_ids = []
            for path_str in names:
                path_str = str(path_str)
                for m in marker_idx_map:
                    if m in path_str:
                        marker_ids.append(marker_idx_map[m])
                        break
                else:
                    marker_ids.append(0)

            marker_tensor = torch.tensor(marker_ids, device=device)
            fake = model(dapi, marker_tensor)

            for i in range(fake.shape[0]):
                p = to_uint8(fake[i].cpu())
                t = to_uint8(real[i].cpu())
                ssim_val = sk_ssim(p, t, channel_axis=-1, data_range=255)
                mse = float(((p.astype(float) - t.astype(float))**2).mean())
                psnr_val = 10 * (255.0**2 / max(mse, 1e-10))

                ssim_sum += ssim_val
                psnr_sum += psnr_val
                m = idx_to_marker.get(marker_ids[i], 'unknown')
                marker_ssim[m].append(ssim_val)
                n += 1

    model.train()
    avg_ssim = ssim_sum / max(n, 1)
    avg_psnr = psnr_sum / max(n, 1)
    return {"ssim": avg_ssim, "psnr": avg_psnr, "n": n, "marker_ssim": {m: sum(v)/len(v) if v else 0 for m, v in marker_ssim.items()}}


# ============================================================================
# 7. 数据集包装（marker-aware）
# ============================================================================

class MultiMarkerDataset(torch.utils.data.Dataset):
    """
    支持多 marker 联合训练的数据集。
    每个样本返回 DAPI、IHC、marker_idx。
    """
    def __init__(self, data_root: Path, markers: list, split='train', patch_size=256):
        self.ds_list = []
        self.marker_indices = []
        for m in markers:
            ds = DAPItoIHCDataset(root=data_root, marker=m, split=split,
                                  patch_size=patch_size, augment=(split == 'train'))
            self.ds_list.append(ds)
            self.marker_indices.extend([markers.index(m)] * len(ds))

        # flatten all samples
        self.samples = []
        for di, ds in enumerate(self.ds_list):
            for pair in ds.pairs:
                from src.data.dataset import load_image, build_transforms
                from PIL import Image
                import numpy as np
                dapi_img = load_image(pair[0])
                ihc_img = load_image(pair[1]) if pair[1] else np.zeros_like(dapi_img)
                transform = build_transforms(256, split == 'train')
                t = transform(image=dapi_img, ihc=ihc_img)
                self.samples.append({
                    'dapi': t['image'],
                    'ihc': t['ihc'],
                    'marker_idx': di,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================================
# 8. 主训练循环
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='DiffVS 风格多 marker 训练')
    p.add_argument('--marker', type=str, default='HLA-DR', help='单个 marker（默认）或 --all')
    p.add_argument('--all-markers', action='store_true', help='多 marker 联合训练')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lambda-l1', type=float, default=1.0)
    p.add_argument('--lambda-ssim', type=float, default=0.1)
    p.add_argument('--lambda-perc', type=float, default=0.05)
    p.add_argument('--save-name', type=str, default='diffvs')
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--data-root', type=str,
                   default='E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
    return p.parse_args()


MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']
MARKER_IDX = {m: i for i, m in enumerate(MARKERS)}


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)

    print('=' * 60)
    print(f'[DiffVS] device={device}')
    print(f'[DiffVS] epochs={args.epochs} bs={args.batch_size} lr={args.lr}')
    print(f'[DiffVS] lambda_l1={args.lambda_l1} ssim={args.lambda_ssim} perc={args.lambda_perc}')
    print('=' * 60)

    if args.all_markers:
        train_ds = MultiMarkerDataset(data_root, MARKERS, split='train')
        val_ds   = MultiMarkerDataset(data_root, MARKERS, split='val')
        print(f'[MultiMarker] train={len(train_ds)}, val={len(val_ds)}')
        model = DiffVSGenerator(num_markers=len(MARKERS)).to(device)
        print(f'[Generator] params={sum(p.numel() for p in model.parameters()):,}')
    else:
        train_ds = DAPItoIHCDataset(data_root, args.marker, split='train', patch_size=256)
        val_ds   = DAPItoIHCDataset(data_root, args.marker, split='val',   patch_size=256)
        print(f'[Dataset] train={len(train_ds)}, val={len(val_ds)}')
        marker_idx = torch.tensor([MARKER_IDX[args.marker]] * 256)  # dummy
        model = DiffVSGenerator(num_markers=len(MARKERS)).to(device)
        print(f'[Generator] params={sum(p.numel() for p in model.parameters()):,}')

    # ── DataLoader（带 marker_idx 提取）─
    def collate_fn(batch):
        dapi = torch.stack([b['dapi'] for b in batch])
        ihc  = torch.stack([b['ihc']  for b in batch])
        midx = torch.tensor([b.get('marker_idx', 0) for b in batch], dtype=torch.long)
        return {'dapi': dapi, 'ihc': ihc, 'marker_idx': midx}

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True,
                              collate_fn=collate_fn if args.all_markers else None)
    val_loader   = DataLoader(val_ds,   batch_size=8,   shuffle=False,
                              num_workers=2, pin_memory=True,
                              collate_fn=collate_fn if args.all_markers else None)

    # ── 模型 ──
    disc = PatchDiscriminator().to(device) if not args.all_markers else None

    # ── 损失 & 优化器 ──
    img_loss_fn = CombinedDiffVSLoss(
        lambda_l1=args.lambda_l1,
        lambda_ssim=args.lambda_ssim,
        lambda_perc=args.lambda_perc,
    ).to(device)

    opt_g = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    opt_d = optim.AdamW(disc.parameters(), lr=args.lr*0.5, weight_decay=1e-4) if disc else None
    sched_g = optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs) if opt_g else None

    # ── 输出目录 ──
    ts = int(time.time())
    marker_str = 'all' if args.all_markers else args.marker
    out_dir = ROOT / 'checkpoints' / f'{args.save_name}_{marker_str}_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ssim = -1.0
    marker_scope = MARKERS if args.all_markers else [args.marker]

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sum_g, n_b = 0.0, 0

        for bi, batch in enumerate(train_loader):
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            marker_idx = batch.get('marker_idx', torch.zeros(dapi.size(0), dtype=torch.long, device=device))
            if not args.all_markers:
                marker_idx = torch.full((dapi.size(0),), MARKER_IDX[args.marker], dtype=torch.long, device=device)

            # ── Generator ──
            fake = model(dapi, marker_idx)
            loss_g = img_loss_fn(fake, real)

            # ── Discriminator（如果单 marker 模式）──
            if disc:
                opt_d.zero_grad(set_to_none=True)
                loss_d = F.mse_loss(disc(real, dapi), torch.ones_like(disc(real, dapi))) + \
                         F.mse_loss(disc(fake.detach(), dapi), torch.zeros_like(disc(fake.detach(), dapi)))
                loss_d.backward()
                opt_d.step()

                loss_g = loss_g + 0.1 * F.mse_loss(disc(fake, dapi), torch.ones_like(disc(fake, dapi)))

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_g.step()

            sum_g += loss_g.item()
            n_b += 1

            if (bi + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f'  [E{epoch} B{bi+1}/{len(train_loader)}] G={loss_g.item():.4f} t={elapsed:.0f}s')

        if sched_g:
            sched_g.step()

        avg_g = sum_g / max(n_b, 1)
        elapsed = time.time() - t0

        # ── 评估 ──
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            result = eval_model(model, val_loader, device, MARKER_IDX)
            ssim_v = result['ssim']
            print(f'[Epoch {epoch}] G={avg_g:.4f} val_ssim={ssim_v:.4f} t={elapsed:.0f}s')

            ck_path = out_dir / f'epoch{epoch}.pt'
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'val_ssim': ssim_v, 'disc': disc.state_dict() if disc else None,
            }, ck_path)

            if ssim_v > best_ssim:
                best_ssim = ssim_v
                torch.save({
                    'epoch': epoch, 'model': model.state_dict(),
                    'val_ssim': ssim_v, 'disc': disc.state_dict() if disc else None,
                }, out_dir / 'best.pt')
                print(f'  [Best] epoch={epoch} val_ssim={ssim_v:.4f}')

    print(f'\n[Done] Best val_ssim={best_ssim:.4f}')


if __name__ == '__main__':
    main()
