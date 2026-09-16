"""条件 UNet：支持时间步 t 和 DAPI 条件图像输入

架构要点：
- x_t（3ch）+ DAPI（3ch）拼接为 6ch 输入
- Sinusoidal 时间嵌入 + AdaGN 全局调制（兼容任意通道数）
- 残差块 + GroupNorm + Self-Attention（仅最深层）
- 输出与 x_t 同维度的速度场 v_θ(x_t, t, x_dapi)

数据流（以 channel_mults=(1,2,4), num_res_blocks=1, 64x64 输入为例）：
  input (64x64) → encoder levels:
    level 0: pool → block → skip (32ch, 32x32)
    level 1: pool → block → skip (64ch, 16x16)
    level 2: (no pool) → block → skip (128ch, 16x16)
  middle (128ch, 16x16)
  decoder levels (顺序 reversed):
    level 2 (no upsample): cat with skip(128ch) → conv → (128ch, 16x16)
    level 1 (upsample): upsample to (64ch, 32x32) → cat with skip(64ch, 16x16) → match size → conv → (64ch, 16x16)
    level 0 (upsample): upsample to (32ch, 32x32) → cat with skip(32ch, 32x32) → conv → (32ch, 32x32)
  output (3ch, 32x32)

注意：由于最后一个 encoder level 不做 pool，decoder 出口的 spatial 会比 input 小一半。
要恢复完整尺寸，最简单的做法：在 decoder 最后一次 upsample 后再 upsample 一次（即使没 skip）。
"""
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    """残差块：支持 x + dapi 拼接输入"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int = 0,
        dropout: float = 0.0,
        use_attention: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.use_attention = use_attention and (out_channels >= 32)
        self.act = SiLU()

        num_groups = max(1, in_channels // 4)
        self.norm1 = nn.GroupNorm(num_groups, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.dapi_proj = nn.Conv2d(cond_channels, out_channels, kernel_size=1) if cond_channels > 0 else None

        num_groups_out = max(1, out_channels // 4)
        self.norm2 = nn.GroupNorm(num_groups_out, out_channels, eps=1e-6)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.attention: Optional[nn.Module] = None
        if self.use_attention:
            nh = max(1, out_channels // 64)
            self.attention = nn.MultiheadAttention(out_channels, nh, batch_first=True, dropout=dropout)
            self.attn_norm = nn.GroupNorm(num_groups_out, out_channels, eps=1e-6)

    def forward(self, x, dapi=None):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        if dapi is not None and self.dapi_proj is not None:
            h = h + self.dapi_proj(dapi)

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        h = h + self.skip(x)

        if self.use_attention and self.attention is not None:
            B, C, H, W = h.shape
            h_t = h.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)
            attn_out, _ = self.attention(h_t, h_t, h_t)
            h = (h_t.squeeze(1) + attn_out.squeeze(1)).reshape(B, H, W, C).permute(0, 3, 1, 2)
            h = self.attn_norm(h)

        return h


class ConditionalUNet(nn.Module):
    """
    条件 UNet，用于 Flow Matching 速度预测 v_θ(x_t, t, x_dapi)

    时间条件通过 AdaGN（Adaptive Group Normalization）注入。
    所有 ada 投影统一输出 ng_max * 2，按当前层 num_groups 取前缀。
    """

    def __init__(
        self,
        in_channels: int = 3,
        cond_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.act = SiLU()

        time_dim = base_channels
        hidden_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_conv = nn.Conv2d(in_channels + cond_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        cur_ch = base_channels
        self.encoder_levels = nn.ModuleList()
        self.num_levels = len(self.channel_mults)
        for i, mult in enumerate(self.channel_mults):
            out_ch = base_channels * mult
            level = nn.ModuleDict({
                "blocks": nn.ModuleList([
                    ResBlock(cur_ch if j == 0 else out_ch, out_ch,
                             cond_channels=cond_channels, dropout=dropout,
                             use_attention=(out_ch >= base_channels * 2 and j == num_res_blocks - 1))
                    for j in range(num_res_blocks)
                ]),
                # 永远 pool；最后的 spatial 缩小由最后一个 level 的 pool 完成
                "pool": nn.MaxPool2d(2),
            })
            self.encoder_levels.append(level)
            cur_ch = out_ch

        # Middle
        mid_ch = cur_ch
        ng_mid = max(1, mid_ch // 4)
        self.mid_gn1 = nn.GroupNorm(ng_mid, mid_ch, eps=1e-6)
        self.mid_conv = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1)
        self.mid_gn2 = nn.GroupNorm(ng_mid, mid_ch, eps=1e-6)

        # Decoder: 与 encoder 一对一镜像，每个 level 处理一组 num_res_blocks 个 skip
        self.decoder_levels = nn.ModuleList()
        ng_max = ng_mid
        dec_cur_ch = mid_ch
        for i in range(self.num_levels):
            enc_level_idx = self.num_levels - 1 - i  # 0=最深, num_levels-1=最浅
            enc_out_ch = base_channels * self.channel_mults[enc_level_idx]
            # concat 后的通道：当前 dec_cur_ch + encoder 对应 level 的 out_ch
            concat_ch = dec_cur_ch + enc_out_ch
            ng_up = max(1, concat_ch // 4)
            ng_max = max(ng_max, ng_up)
            self.decoder_levels.append(nn.ModuleDict({
                "gn": nn.GroupNorm(ng_up, concat_ch, eps=1e-6),
                "conv": nn.Conv2d(concat_ch, enc_out_ch, kernel_size=3, padding=1),
            }))
            dec_cur_ch = enc_out_ch

        self.ada_mid1 = nn.Linear(hidden_dim, ng_max * 2)
        self.ada_mid2 = nn.Linear(hidden_dim, ng_max * 2)
        self.ada_out = nn.Linear(hidden_dim, ng_max * 2)

        # Output
        ng_out = max(1, dec_cur_ch // 4)
        ng_max = max(ng_max, ng_out)
        # Recompute ada layers if ng_max grew
        # (上面已用 max; 若 ng_out > ng_max，需要更新)
        if ng_out * 2 > self.ada_out.out_features:
            self.ada_mid1 = nn.Linear(hidden_dim, ng_out * 2)
            self.ada_mid2 = nn.Linear(hidden_dim, ng_out * 2)
            self.ada_out = nn.Linear(hidden_dim, ng_out * 2)
        self.out_gn = nn.GroupNorm(ng_out, dec_cur_ch, eps=1e-6)
        self.out_conv = nn.Conv2d(dec_cur_ch, out_channels, kernel_size=3, padding=1)
        self.out_conv.weight.data.zero_()
        self.out_conv.bias.data.zero_()

    def _ada_gn(self, h, gn, ada_proj):
        """AdaGN: GroupNorm + adaptive scale/shift from time embedding"""
        h_norm = gn(h)
        ng = gn.num_groups
        scale_shift = ada_proj(self.t_emb)[:, :ng * 2]
        scale, shift = scale_shift.chunk(2, dim=-1)
        C = h_norm.shape[1]
        scale = scale[:, :, None, None].repeat(1, C // ng, h_norm.shape[2], h_norm.shape[3])
        shift = shift[:, :, None, None].repeat(1, C // ng, h_norm.shape[2], h_norm.shape[3])
        return h_norm * (scale + 1.0) + shift

    def forward(self, x: torch.Tensor, t: torch.Tensor, dapi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:   当前噪声状态 (B, 3, H, W)
            t:   时间步 (B,)，范围 [0,1]
            dapi: DAPI 条件图 (B, 3, H, W)
        Returns:
            v:   预测速度场 (B, 3, H, W)
        """
        # 保存原始 input 尺寸（H, W），后续上采样对齐使用
        self._input_hw = (x.shape[-2], x.shape[-1])
        self.t_emb = self.time_mlp(timestep_embedding(t, self.time_mlp[0].in_features))

        if dapi is not None:
            x = torch.cat([x, dapi], dim=1)

        h = self.input_conv(x)

        # Encoder: 收集每层的全部 skip（num_res_blocks 个/层）
        skips_per_level: list[list[torch.Tensor]] = [[] for _ in range(self.num_levels)]
        for level_idx, level in enumerate(self.encoder_levels):
            h = level["pool"](h)
            for block in level["blocks"]:
                if dapi is not None:
                    dapi_cur = F.interpolate(dapi, size=h.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    dapi_cur = None
                h = block(h, dapi_cur)
                skips_per_level[level_idx].append(h)

        # Middle
        h = self._ada_gn(h, self.mid_gn1, self.ada_mid1)
        h = self.act(h)
        h = self.mid_conv(h)
        h = self._ada_gn(h, self.mid_gn2, self.ada_mid2)
        h = self.act(h)

        # Decoder: 倒序遍历 encoder levels（从最深到最浅）
        # 每次都强制 upsample 到 skip 的 spatial 尺寸（若已匹配则不缩放）
        for di, level in enumerate(self.decoder_levels):
            enc_level_idx = self.num_levels - 1 - di
            # 使用该 encoder level 的最后一个 skip
            skip = skips_per_level[enc_level_idx][-1]
            # 强制 upsample 到 input 的尺寸（与原图保持一致）
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            # concat
            h = torch.cat([h, skip], dim=1)
            # AdaGN
            ng_cur = level["gn"].num_groups
            scale_shift = self.ada_out(self.t_emb)[:, :ng_cur * 2]
            scale, shift = scale_shift.chunk(2, dim=-1)
            C = h.shape[1]
            scale = scale[:, :, None, None].repeat(1, C // ng_cur, h.shape[2], h.shape[3])
            shift = shift[:, :, None, None].repeat(1, C // ng_cur, h.shape[2], h.shape[3])
            h = level["gn"](h) * (scale + 1.0) + shift
            h = self.act(h)
            h = level["conv"](h)

        # 最终上采样回到原始 input 尺寸（不依赖后续 x 的形状）
        # 由于 forward 开头对 x 与 dapi 做拼接后 x.shape 已经变化，这里在 forward 入口保存原始 H/W
        if h.shape[-2:] != self._input_hw:
            h = F.interpolate(h, size=self._input_hw, mode="bilinear", align_corners=False)

        # Output
        h = self._ada_gn(h, self.out_gn, self.ada_out)
        h = self.act(h)
        h = self.out_conv(h)
        return h
