"""
DiffVS (AAAI 2026) 风格虚拟染色方案 - 重构版

核心设计：
- Stage 1: DAPI encoder + marker embedding + decoder 联合训练
- Stage 2: Latent Flow Matching 微调
- 单步推理：直接通过 Flow Matching 采样

输入: DAPI (256x256, 3ch)
输出: IHC (256x256, 3ch)
条件: marker embedding (4 markers)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import builtins
_oi = builtins.print
def _fp(*a, **k):
    k.setdefault('flush', True)
    try:
        _oi(*a, **k)
    except UnicodeEncodeError:
        safe = [str(x).encode('gbk', errors='replace').decode('gbk') for x in a]
        _oi(*safe, **k)
builtins.print = _fp

from src.data.dataset import DAPItoIHCDataset
from src.metrics.ssim_psnr import to_uint8


# ============================================================================
# 核心模块
# ============================================================================

class MarkerEmbedding(nn.Module):
    """Marker 条件嵌入"""
    def __init__(self, num_markers: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_markers, embed_dim)

    def forward(self, marker_idx: torch.Tensor) -> torch.Tensor:
        return self.embedding(marker_idx)


class SimpleGate(nn.Module):
    """NAFNet SimpleGate"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=1)
        return x * torch.sigmoid(gate)


class ResBlock(nn.Module):
    """残差块 + 条件调制"""
    def __init__(self, ch: int, emb_ch: int = 0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(4, ch)
        self.conv1 = nn.Conv2d(ch, ch * 4, 1)
        self.gate = SimpleGate()
        self.conv2 = nn.Conv2d(ch * 2, ch, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Identity()
        self.emb_ch = emb_ch
        if emb_ch > 0:
            self.ada_proj = nn.Linear(emb_ch, ch * 2)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm(x)
        h = self.conv1(h)
        h = self.gate(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.emb_ch > 0 and emb is not None:
            scale, shift = self.ada_proj(emb).chunk(2, dim=-1)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + scale) + shift
        return x + self.skip(h)


# ============================================================================
# 主模型
# ============================================================================

class DiffVSNet(nn.Module):
    """
    DiffVS 主模型: DAPI -> IHC
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 3,
                 base_ch: int = 64, num_markers: int = 4, marker_dim: int = 32):
        super().__init__()
        self.base_ch = base_ch
        self.marker_dim = marker_dim

        # Marker embedding
        self.marker_embed = MarkerEmbedding(num_markers, marker_dim)

        # 输入投影
        self.input_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)

        # Encoder (下采样)
        self.enc1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            nn.GroupNorm(4, base_ch * 2),
            nn.SiLU(),
            ResBlock(base_ch * 2, marker_dim)
        )  # 128ch

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
            ResBlock(base_ch * 4, marker_dim),
            ResBlock(base_ch * 4, marker_dim)
        )  # 256ch

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1),
            nn.GroupNorm(8, base_ch * 8),
            nn.SiLU(),
            ResBlock(base_ch * 8, marker_dim),
            ResBlock(base_ch * 8, marker_dim),
            ResBlock(base_ch * 8, marker_dim)
        )  # 512ch (latent)

        # Decoder (上采样 + skip)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, 2),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
            ResBlock(base_ch * 4, marker_dim),
            ResBlock(base_ch * 4, marker_dim)
        )  # 256ch

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 8, base_ch * 2, 2, 2),  # skip 后 256+256=512 -> 256
            nn.GroupNorm(4, base_ch * 2),
            nn.SiLU(),
            ResBlock(base_ch * 2, marker_dim),
            ResBlock(base_ch * 2, marker_dim)
        )  # 128ch

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch, 2, 2),  # skip 后 128+128=256 -> 64
            nn.GroupNorm(4, base_ch),
            nn.SiLU(),
            ResBlock(base_ch, marker_dim),
            ResBlock(base_ch, marker_dim)
        )  # 64ch

        # 输出
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, marker_idx: torch.Tensor) -> torch.Tensor:
        emb = self.marker_embed(marker_idx)

        # Encode
        e1 = self.input_conv(x)  # 64ch
        e2 = self.enc1(e1)       # 128ch
        e3 = self.enc2(e2)       # 256ch
        e4 = self.enc3(e3)       # 512ch

        # Decode
        d3 = self.dec3(e4)       # 256ch
        d3 = torch.cat([d3, e3], dim=1)  # 256+256=512

        d2 = self.dec2(d3)        # 128ch
        d2 = torch.cat([d2, e2], dim=1)  # 128+128=256

        d1 = self.dec1(d2)       # 64ch
        d1 = torch.cat([d1, e1], dim=1)  # 64+64=128

        return self.out_conv(d1)


class LatentFlowMatching(nn.Module):
    """隐空间 Flow Matching"""
    def __init__(self, latent_ch: int, marker_dim: int, time_dim: int = 64):
        super().__init__()
        self.latent_ch = latent_ch
        self.marker_dim = marker_dim

        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, latent_ch),
            nn.SiLU(),
            nn.Linear(latent_ch, latent_ch)
        )

        # Marker 投影到 latent_ch
        self.marker_proj = nn.Sequential(
            nn.Linear(marker_dim, latent_ch),
            nn.SiLU()
        )

        # 速度预测
        hidden = latent_ch * 2
        self.net = nn.Sequential(
            nn.Conv2d(latent_ch * 2, hidden, 3, 1, 1),  # z_t + marker_cond
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, latent_ch, 3, 1, 1)
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def timestep_emb(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, z_t: torch.Tensor, marker_emb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z_t.shape
        # 条件投影
        m_cond = self.marker_proj(marker_emb).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        t_cond = self.time_mlp(self.timestep_emb(t, 64)).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined = torch.cat([z_t, m_cond + t_cond], dim=1)
        return self.net(combined)


class DiffVSFull(nn.Module):
    """
    完整 DiffVS: Stage 1 (直接回归) + Stage 2 (Latent FM)
    """
    def __init__(self, in_ch: int = 3, out_ch: int = 3,
                 base_ch: int = 64, num_markers: int = 4, marker_dim: int = 32):
        super().__init__()
        self.base_ch = base_ch

        # Stage 1: 直接回归模型
        self.main = DiffVSNet(in_ch, out_ch, base_ch, num_markers, marker_dim)

        # Stage 2: Latent FM
        self.latent_fm = LatentFlowMatching(base_ch * 8, marker_dim)

    def forward_stage1(self, x: torch.Tensor, marker_idx: torch.Tensor) -> torch.Tensor:
        return self.main(x, marker_idx)

    def _encode(self, x: torch.Tensor, marker_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list]:
        """提取 latent 特征"""
        emb = self.main.marker_embed(marker_idx)
        e1 = self.main.input_conv(x)
        e2 = self.main.enc1(e1)
        e3 = self.main.enc2(e2)
        e4 = self.main.enc3(e3)
        return e4, emb, [e1, e2, e3]

    def forward_train_fm(self, x: torch.Tensor, target: torch.Tensor,
                        marker_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow Matching 训练"""
        B = x.shape[0]
        device = x.device

        # 获取目标 latent
        with torch.no_grad():
            z0, _, _ = self._encode(target, marker_idx)

        # 获取 x 的 latent 作为条件
        z_dapi, marker_emb, skips = self._encode(x, marker_idx)

        # Flow Matching
        t = torch.rand(B, device=device)
        epsilon = torch.randn_like(z0)
        z_t = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * epsilon
        v_target = z0 - epsilon

        # 预测
        v_pred = self.latent_fm(z_t, marker_emb, t)
        loss = torch.mean((v_pred - v_target) ** 2)

        return loss, z_t

    def sample(self, x: torch.Tensor, marker_idx: torch.Tensor,
               num_steps: int = 20, solver: str = 'euler') -> torch.Tensor:
        """Flow Matching 采样"""
        B, _, H, W = x.shape
        device = x.device

        # 获取条件
        z_dapi, marker_emb, skips = self._encode(x, marker_idx)

        # 从噪声开始
        z = torch.randn_like(z_dapi)

        dt = 1.0 / num_steps

        if solver == 'heun':
            for i in range(num_steps):
                t_now = i / num_steps
                t_tensor = torch.full((B,), t_now, device=device)
                v1 = self.latent_fm(z, marker_emb, t_tensor)
                z_mid = z + dt * v1

                t_next = (i + 1) / num_steps
                t_tensor_next = torch.full((B,), t_next, device=device)
                v2 = self.latent_fm(z_mid, marker_emb, t_tensor_next)
                z = z + dt * 0.5 * (v1 + v2)
        else:
            for i in range(num_steps):
                t_now = i / num_steps
                t_tensor = torch.full((B,), t_now, device=device)
                v = self.latent_fm(z, marker_emb, t_tensor)
                z = z + dt * v

        # 解码
        emb = marker_emb
        d3 = self.main.dec3(z)
        d3 = torch.cat([d3, skips[2]], dim=1)
        d2 = self.main.dec2(d3)
        d2 = torch.cat([d2, skips[1]], dim=1)
        d1 = self.main.dec1(d2)
        d1 = torch.cat([d1, skips[0]], dim=1)
        return self.main.out_conv(d1)


# ============================================================================
# 损失函数
# ============================================================================

class SSIMLoss(nn.Module):
    def __init__(self, channel: int = 3, window_size: int = 11):
        super().__init__()
        self.channel = channel
        self.window_size = window_size
        self._init_window()

    def _init_window(self):
        sigma = 1.5
        gauss = torch.tensor([
            torch.exp(torch.tensor(-(x - self.window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(self.window_size)
        ])
        gauss /= gauss.sum()
        g2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        self.register_buffer('win', g2d.view(1, 1, self.window_size, self.window_size).expand(
            self.channel, 1, self.window_size, self.window_size).contiguous())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        w = self.win.to(pred.device)
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, w, padding=pad, groups=self.channel)
        mu_target = F.conv2d(target, w, padding=pad, groups=self.channel)
        mu_pred_sq, mu_target_sq = mu_pred ** 2, mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred = F.conv2d(pred ** 2, w, padding=pad, groups=self.channel) - mu_pred_sq
        sigma_target = F.conv2d(target ** 2, w, padding=pad, groups=self.channel) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, w, padding=pad, groups=self.channel) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred + sigma_target + C2))
        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    def __init__(self, lambda_l1: float = 1.0, lambda_ssim: float = 0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(pred, target) * self.lambda_l1 + self.ssim(pred, target) * self.lambda_ssim


# ============================================================================
# 数据集
# ============================================================================

def collate_batch(batch, marker_idx: int = 0):
    return {
        'dapi': torch.stack([b['dapi'] for b in batch]),
        'ihc': torch.stack([b['ihc'] for b in batch]),
        'marker_idx': torch.tensor([b.get('marker_idx', marker_idx) for b in batch], dtype=torch.long)
    }


# ============================================================================
# 评估
# ============================================================================

def eval_model(model: nn.Module, loader: DataLoader, device: torch.device,
               stage: str = 'stage1', num_steps: int = 20) -> dict:
    model.eval()
    ssim_sum, psnr_sum, n = 0.0, 0.0, 0

    with torch.inference_mode():
        for batch in loader:
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            marker_idx = batch['marker_idx'].to(device)

            if stage == 'fm':
                fake = model.sample(dapi, marker_idx, num_steps=num_steps)
            else:
                fake = model.forward_stage1(dapi, marker_idx)

            for i in range(fake.shape[0]):
                p = to_uint8(fake[i].cpu())
                t = to_uint8(real[i].cpu())
                ssim_sum += sk_ssim(p, t, channel_axis=-1, data_range=255)
                mse = float(((p.astype(np.float32) - t.astype(np.float32)) ** 2).mean())
                psnr_sum += 10 * np.log10((255.0 ** 2) / max(mse, 1e-10))
                n += 1

    model.train()
    return {'ssim': ssim_sum / n, 'psnr': psnr_sum / n, 'n': n}


# ============================================================================
# 训练
# ============================================================================

def main():
    p = argparse.ArgumentParser(description='DiffVS 虚拟染色')
    p.add_argument('--marker', type=str, default='HLA-DR')
    p.add_argument('--stage', type=str, default='stage1', choices=['stage1', 'fm', 'joint'])
    p.add_argument('--stage1_ckpt', type=str, default='')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--base_ch', type=int, default=64)
    p.add_argument('--marker_dim', type=int, default=32)
    p.add_argument('--num_steps', type=int, default=20)
    p.add_argument('--save_name', type=str, default='diffvs_v2')
    p.add_argument('--data_root', type=str,
                   default='E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）')
    p.add_argument('--use_val_split', action='store_true')
    args = p.parse_args()
    # 支持 IHC_DATA_ROOT 环境变量（避免 PowerShell 传中文 argv 时的编码陷阱）
    import os as _os
    env_root = _os.environ.get('IHC_DATA_ROOT')
    if env_root:
        args.data_root = env_root

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MARKERS = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']
    marker_idx = MARKERS.index(args.marker)
    data_root = Path(args.data_root)

    if args.use_val_split:
        train_data_root = data_root / 'train_subset'
        val_data_root = data_root  # val 在原始 data_root 下
    else:
        train_data_root = data_root
        val_data_root = data_root

    print('=' * 60)
    print(f'[DiffVS] marker={args.marker} stage={args.stage} epochs={args.epochs}')
    print(f'[DiffVS] bs={args.batch_size} lr={args.lr} num_steps={args.num_steps}')
    print(f'[DiffVS] device={device}')
    print('=' * 60)

    train_ds = DAPItoIHCDataset(train_data_root, args.marker, split='train', patch_size=256)
    val_ds = DAPItoIHCDataset(val_data_root, args.marker, split='val', patch_size=256)
    print(f'[Data] train={len(train_ds)}, val={len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=True,
                              collate_fn=lambda b: collate_batch(b, marker_idx))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=0, pin_memory=True,
                            collate_fn=lambda b: collate_batch(b, marker_idx))

    model = DiffVSFull(
        base_ch=args.base_ch,
        num_markers=len(MARKERS),
        marker_dim=args.marker_dim
    ).to(device)
    print(f'[Model] params={sum(p.numel() for p in model.parameters()):,}')

    if args.stage1_ckpt:
        print(f'[Load] {args.stage1_ckpt}')
        ckpt = torch.load(args.stage1_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)

    if args.stage == 'stage1':
        loss_fn = CombinedLoss(lambda_l1=1.0, lambda_ssim=0.1).to(device)
        train_params = list(model.parameters())
    elif args.stage == 'fm':
        loss_fn = nn.MSELoss().to(device)
        train_params = list(model.latent_fm.parameters())
        for p in model.main.parameters():
            p.requires_grad = False
        print('[Stage 2] Frozen main, training Latent FM')
    else:
        loss_fn = CombinedLoss(lambda_l1=1.0, lambda_ssim=0.1).to(device)
        train_params = list(model.parameters())

    print(f'[Trainable] {sum(p.numel() for p in train_params):,} params')

    opt = optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    ts = int(time.time())
    out_dir = ROOT / 'checkpoints' / f'{args.save_name}_{args.marker}_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    init_metrics = eval_model(model, val_loader, device, stage=args.stage)
    print(f'[Init] val_ssim={init_metrics["ssim"]:.4f} val_psnr={init_metrics["psnr"]:.2f}')

    best_ssim = init_metrics['ssim']
    best_epoch = -1

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum, n_b = 0.0, 0

        for bi, batch in enumerate(train_loader):
            dapi = batch['dapi'].to(device)
            real = batch['ihc'].to(device)
            midx = batch['marker_idx'].to(device)

            opt.zero_grad(set_to_none=True)

            if args.stage == 'stage1':
                fake = model.forward_stage1(dapi, midx)
                loss = loss_fn(fake, real)
            elif args.stage == 'fm':
                loss, _ = model.forward_train_fm(dapi, real, midx)
            else:
                fake = model.forward_stage1(dapi, midx)
                loss_l1 = loss_fn(fake, real)
                loss_fm, _ = model.forward_train_fm(dapi, real, midx)
                loss = loss_l1 + loss_fm

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += loss.item()
            n_b += 1

            if (bi + 1) % 50 == 0:
                print(f'  [E{ep} B{bi+1}/{len(train_loader)}] L={loss.item():.4f} t={time.time()-t0:.0f}s')

        sched.step()
        elapsed = time.time() - t0

        eval_stage = 'fm' if args.stage == 'fm' else 'stage1'
        metrics = eval_model(model, val_loader, device, stage=eval_stage)

        print(f'[Epoch {ep}] L={loss_sum/max(n_b,1):.4f} '
              f'val_ssim={metrics["ssim"]:.4f} val_psnr={metrics["psnr"]:.2f} t={elapsed:.0f}s')

        ck = {'epoch': ep, 'model': model.state_dict(), **metrics}
        torch.save(ck, out_dir / f'epoch{ep}.pt')

        if metrics['ssim'] > best_ssim:
            best_ssim = metrics['ssim']
            best_epoch = ep
            torch.save(ck, out_dir / 'best.pt')
            print(f'  [Best] epoch={ep} ssim={best_ssim:.4f}')

    print(f'\n[Done] Best val_ssim={best_ssim:.4f} (epoch {best_epoch})')


if __name__ == '__main__':
    main()
