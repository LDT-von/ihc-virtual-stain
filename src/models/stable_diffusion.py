"""
简化版 Stable Diffusion 风格的虚拟染色模型

直接在像素空间做 DDPM/DDIM 扩散，不依赖 VAE

Author: AI Assistant
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


# ============================================================================
# 辅助模块
# ============================================================================

def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal timestep embedding"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb


class ResBlock(nn.Module):
    """残差块 + 条件调制"""
    def __init__(self, channels: int, time_ch: int = 0, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        
        # 时间步和 DAPI 条件
        if time_ch > 0:
            self.time_emb = nn.Sequential(
                nn.Linear(time_ch, channels * 2),
            )
        
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        if cond is not None and hasattr(self, 'time_emb'):
            scale_shift = self.time_emb(cond)  # (B, channels*2)
            scale, shift = scale_shift[:, :self.channels], scale_shift[:, self.channels:]
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            h = h * (scale + 1) + shift
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return x + h


class Attention(nn.Module):
    """简化的自注意力"""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).chunk(3, dim=1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q.reshape(B, C, H * W)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W)
        
        attn = torch.bmm(q.transpose(1, 2), k) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        h = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        
        return x + self.proj(h)


class SimpleDiffusionUNet(nn.Module):
    """
    简化的 UNet 用于像素空间扩散
    """
    def __init__(
        self,
        in_channels: int = 3,  # RGB
        cond_channels: int = 3,  # DAPI
        base_ch: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 时间嵌入
        hidden_dim = base_ch * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # DAPI 编码器
        self.dapi_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dapi_proj = nn.Linear(128, hidden_dim)
        
        # 输入卷积
        self.input_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        
        # Encoder
        self.enc1 = ResBlock(base_ch, hidden_dim, dropout)
        self.enc1_down = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        
        self.enc2 = ResBlock(base_ch * 2, hidden_dim, dropout)
        self.enc2_down = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        
        self.enc3 = ResBlock(base_ch * 4, hidden_dim, dropout)
        self.enc3_down = nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1)
        
        self.enc4 = ResBlock(base_ch * 8, hidden_dim, dropout)
        self.enc4_down = nn.Conv2d(base_ch * 8, base_ch * 8, 3, stride=2, padding=1)
        
        # Middle
        self.mid = ResBlock(base_ch * 8, hidden_dim, dropout)
        self.mid_attn = Attention(base_ch * 8)
        self.mid2 = ResBlock(base_ch * 8, hidden_dim, dropout)
        
        # Decoder
        # h4: base*8 @ 32x32
        self.dec4_up = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 4, stride=2, padding=1)
        self.dec4_fuse = ResBlock(base_ch * 16, hidden_dim, dropout)  # cat [base*8 + base*8] -> base*16
        self.dec4_reduce = nn.Conv2d(base_ch * 16, base_ch * 8, 1)
        self.dec4 = ResBlock(base_ch * 8, hidden_dim, dropout)

        # h3: base*4 @ 64x64
        self.dec3_up = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1)
        self.dec3_fuse = ResBlock(base_ch * 8, hidden_dim, dropout)
        self.dec3_reduce = nn.Conv2d(base_ch * 8, base_ch * 4, 1)
        self.dec3 = ResBlock(base_ch * 4, hidden_dim, dropout)

        # h2: base*2 @ 128x128
        self.dec2_up = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2_fuse = ResBlock(base_ch * 4, hidden_dim, dropout)
        self.dec2_reduce = nn.Conv2d(base_ch * 4, base_ch * 2, 1)
        self.dec2 = ResBlock(base_ch * 2, hidden_dim, dropout)

        # h1: base @ 256x256
        self.dec1_up = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1_fuse = ResBlock(base_ch * 2, hidden_dim, dropout)
        self.dec1_reduce = nn.Conv2d(base_ch * 2, base_ch, 1)
        self.dec1 = ResBlock(base_ch, hidden_dim, dropout)
        
        # 输出
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_channels, 3, padding=1)
        self.out_conv.weight.data.zero_()
        self.out_conv.bias.data.zero_()
        
        self.base_ch = base_ch
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor, dapi: torch.Tensor) -> torch.Tensor:
        # 时间嵌入
        t_emb = timestep_embedding(t, self.base_ch)
        t_emb = self.time_mlp(t_emb)
        
        # DAPI 条件
        d_emb = self.dapi_encoder(dapi).flatten(1)
        d_emb = self.dapi_proj(d_emb)
        cond = t_emb + d_emb  # (B, hidden_dim)
        
        # Encoder
        h = self.input_conv(x)
        
        h1 = self.enc1(h, cond)
        h = self.enc1_down(h1)
        
        h2 = self.enc2(h, cond)
        h = self.enc2_down(h2)
        
        h3 = self.enc3(h, cond)
        h = self.enc3_down(h3)
        
        h4 = self.enc4(h, cond)
        h = self.enc4_down(h4)
        
        # Middle
        h = self.mid(h, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)

        # Decoder (从最深层向上)
        # h 当前: base*8 @ 16x16, h4: base*8 @ 32x32
        h = self.dec4_up(h)  # base*8 @ 32x32
        h = self.dec4_fuse(torch.cat([h, h4], dim=1))  # base*16
        h = self.dec4_reduce(h)  # base*8
        h = self.dec4(h, cond)

        # h 当前: base*8 @ 32x32 -> 上采样到 64x64 + h3 (base*4)
        h = self.dec3_up(h)  # base*4 @ 64x64
        h = self.dec3_fuse(torch.cat([h, h3], dim=1))  # base*8
        h = self.dec3_reduce(h)  # base*4
        h = self.dec3(h, cond)

        # h 当前: base*4 @ 64x64 -> 上采样到 128x128 + h2 (base*2)
        h = self.dec2_up(h)  # base*2 @ 128x128
        h = self.dec2_fuse(torch.cat([h, h2], dim=1))  # base*4
        h = self.dec2_reduce(h)  # base*2
        h = self.dec2(h, cond)

        # h 当前: base*2 @ 128x128 -> 上采样到 256x256 + h1 (base)
        h = self.dec1_up(h)  # base @ 256x256
        h = self.dec1_fuse(torch.cat([h, h1], dim=1))  # base*2
        h = self.dec1_reduce(h)  # base
        h = self.dec1(h, cond)
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h


# ============================================================================
# 主模型
# ============================================================================

@dataclass
class SimpleDiffusionConfig:
    """简化扩散模型配置"""
    model_channels: int = 128
    dropout: float = 0.1
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_timesteps: int = 1000
    num_sampling_steps: int = 50


class SimpleVirtualStainDiffusion(nn.Module):
    """
    简化虚拟染色扩散模型
    
    直接在像素空间做扩散，不依赖 VAE
    """
    
    def __init__(self, config: SimpleDiffusionConfig):
        super().__init__()
        self.config = config
        
        # UNet (预测噪声)
        self.unet = SimpleDiffusionUNet(
            in_channels=3,
            cond_channels=3,
            base_ch=config.model_channels,
            dropout=config.dropout,
        )
        
        self.num_timesteps = config.num_timesteps
        self.register_schedule()

    def register_schedule(self):
        """注册噪声调度表"""
        config = self.config
        
        betas = torch.linspace(config.beta_start ** 0.5, config.beta_end ** 0.5, config.num_timesteps) ** 2
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向扩散"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        device = x0.device
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(device)[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(device)[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def forward_train(self, x0: torch.Tensor, dapi: torch.Tensor, 
                      cond_drop_prob: float = 0.1) -> torch.Tensor:
        """训练前向传播"""
        B = x0.shape[0]
        device = x0.device
        
        # 时间步
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        noise = torch.randn_like(x0)
        
        # 前向扩散
        x_t = self.q_sample(x0, t, noise)
        
        # CFG: 随机丢弃条件
        if self.training and cond_drop_prob > 0:
            mask = torch.rand(B, device=device) < cond_drop_prob
            dapi_cond = dapi.clone()
            dapi_cond[mask] = 0.0
        else:
            dapi_cond = dapi
        
        # UNet 预测噪声
        noise_pred = self.unet(x_t, t.float() / self.num_timesteps, dapi_cond)
        
        return F.mse_loss(noise_pred, noise, reduction='mean')

    @torch.no_grad()
    def sample_ddim(self, dapi: torch.Tensor, shape: Tuple[int, ...],
                    num_steps: Optional[int] = None) -> torch.Tensor:
        """DDIM 采样"""
        num_steps = num_steps or self.config.num_sampling_steps
        device = dapi.device
        B = shape[0]
        
        step_size = self.num_timesteps // num_steps
        times = torch.arange(0, self.num_timesteps, step_size, device=device).flip(0)
        
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(times):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            t_norm = t_tensor.float() / self.num_timesteps
            
            # 预测噪声
            noise_pred = self.unet(x, t_norm, dapi)
            
            # 预测 x0
            alpha_t = self.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            if i < len(times) - 1:
                next_t = times[i + 1]
                alpha_next = self.alphas_cumprod[next_t]
                
                # DDIM 步
                pred_dir = torch.sqrt(1 - alpha_next) * noise_pred
                x = torch.sqrt(alpha_next) * x0_pred + pred_dir
            else:
                x = x0_pred
        
        return torch.tanh(x)

    @torch.no_grad()
    def sample(self, dapi: torch.Tensor, shape: Tuple[int, ...],
               num_steps: Optional[int] = None,
               method: Optional[str] = None) -> torch.Tensor:
        """主采样接口"""
        return self.sample_ddim(dapi, shape, num_steps)


# ============================================================================
# 模型构建
# ============================================================================

def build_simple_stain_sd(
    model_channels: int = 128,
    dropout: float = 0.1,
    num_sampling_steps: int = 50,
) -> SimpleVirtualStainDiffusion:
    """构建简化虚拟染色 SD 模型"""
    config = SimpleDiffusionConfig(
        model_channels=model_channels,
        dropout=dropout,
        num_sampling_steps=num_sampling_steps,
    )
    return SimpleVirtualStainDiffusion(config)


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    config = SimpleDiffusionConfig(model_channels=64)
    model = SimpleVirtualStainDiffusion(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    x = torch.randn(2, 3, 256, 256).to(device)
    dapi = torch.randn(2, 3, 256, 256).to(device)
    
    loss = model.forward_train(x, dapi)
    print(f"Loss: {loss.item():.4f}")
    
    gen = model.sample(dapi, (2, 3, 256, 256))
    print(f"Generated: {gen.shape}")
    print("Smoke test passed!")
