"""Pix2Pix GAN: U-Net Generator + Patch Discriminator

核心组件：
- U-Net Generator: 编码器-解码器结构，带跳跃连接
- Patch Discriminator: Patch-wise 判别器，用于 LSGAN 训练
- 条件 GAN 框架：DAPI 作为条件输入

参考：Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (CVPR 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """卷积块：Conv -> BN -> LeakyReLU"""
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """残差块：用于 U-Net 编码器/解码器深层"""
    def __init__(self, channels: int, dropout: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AttentionBlock(nn.Module):
    """自注意力块：用于捕获长距离依赖"""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        
        # Query, Key, Value
        f = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C'
        g = self.key(x).view(B, -1, H * W)  # B, C', HW
        h = self.value(x).view(B, -1, H * W).permute(0, 2, 1)  # B, HW, C
        
        # Attention
        s = torch.bmm(f, g)  # B, HW, HW
        attention = F.softmax(s, dim=-1)
        
        o = torch.bmm(attention, h)  # B, HW, C
        o = o.permute(0, 2, 1).contiguous().view(B, C, H, W)
        
        return self.gamma * o + x


class UNetGenerator(nn.Module):
    """
    U-Net Generator
    
    编码器：下采样 5 次，最终特征图尺寸为 8x8
    解码器：上采样 5 次，恢复原始尺寸
    
    特点：
    - DAPI 与输入图像在第一层拼接
    - 编码器和解码器之间使用残差连接
    - 可选的自注意力机制
    """
    def __init__(
        self,
        input_channels: int = 3,      # 输入图像通道数
        cond_channels: int = 3,         # 条件图像（DAPI）通道数
        output_channels: int = 3,       # 输出图像通道数
        base_filters: int = 64,        # 基础滤波器数量
        use_attention: bool = True,    # 是否使用注意力机制
        use_residual: bool = True,     # 是否使用残差连接
        dropout: float = 0.5,          # Dropout 概率
    ):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # 初始卷积：DAPI 与输入图像拼接
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels + cond_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 编码器（下采样）- 减少到 5 层
        # 输入: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.ModuleList([
            ConvBlock(64, 128),        # 128
            ConvBlock(128, 256),        # 64
            ConvBlock(256, 512),        # 32
            ConvBlock(512, 512),        # 16
            ConvBlock(512, 512),        # 8
        ])
        
        # 中间层（最深层）- 添加更多残差块
        self.middle = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512, dropout),
            ResidualBlock(512, dropout),
        )
        
        # 注意力模块
        if use_attention:
            self.attention = AttentionBlock(512)
        
        # 解码器（上采样）- 5 层
        # 输出: 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
            ),  # 16
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
            ),  # 32
            nn.Sequential(
                nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),  # 64
            nn.Sequential(
                nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),  # 128
            nn.Sequential(
                nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            ),  # 256
        ])
        
        # 最终输出层 - 使用双线性插值 + 卷积来恢复尺寸
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )
    
    def forward(self, x: torch.Tensor, dapi: Optional[torch.Tensor] = None, return_feats: bool = False) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, 3, H, W)
            dapi: 条件图像 (B, 3, H, W)
            return_feats: 是否同时返回 encoder 第 1 层特征（64 通道）
        Returns:
            生成的图像 (B, 3, H, W)；若 return_feats=True 则 (img, feat)
            feat 形状 (B, 64, H/2, W/2)
        """
        # 初始层：将输入与 DAPI 拼接
        if dapi is not None:
            x = torch.cat([x, dapi], dim=1)
        x = self.input_conv(x)

        # 编码器：保存跳跃连接
        encoder_outputs = [x]
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            encoder_outputs.append(x)
            if return_feats and i == 0:
                # encoder[0] 输出 64 通道 (B, 64, H/2, W/2)
                feat_for_vsmt = x
        
        # 中间层
        x = self.middle(x)
        
        # 注意力机制
        if self.use_attention:
            x = self.attention(x)
        
        # 解码器：使用跳跃连接
        # 5 个解码器层对应 5 个编码器层（从深到浅）
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            # 跳跃连接：编码器第 4-i 层（从最深层到最浅层）
            skip_idx = 4 - i
            x = torch.cat([x, encoder_outputs[skip_idx]], dim=1)
        
        # 最终上采样恢复尺寸
        x = self.upsample(x)
        x = self.final_conv(x)

        if return_feats:
            return x, feat_for_vsmt
        return x


class PatchDiscriminator(nn.Module):
    """
    Patch Discriminator (PatchGAN)
    
    判别器在图像的多个重叠 patch 上进行判断，
    可以更好地捕获高频结构和纹理细节
    """
    def __init__(
        self,
        input_channels: int = 3,
        cond_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        
        # 初始层：DAPI 与输入图像拼接
        self.input_conv = nn.Conv2d(input_channels + cond_channels, ndf, kernel_size=4, stride=2, padding=1)
        self.input_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # 中间层
        layers = []
        prev_channels = ndf
        for i in range(1, n_layers + 1):
            out_channels = min(ndf * (2 ** i), 512)
            stride = 1 if i == n_layers else 2
            layers.append(
                nn.Conv2d(prev_channels, out_channels, kernel_size=4, stride=stride, padding=1)
            )
            if i < n_layers:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channels = out_channels
        
        self.middle = nn.Sequential(*layers)
        
        # 最终层：输出 patch 级别的预测
        self.output_conv = nn.Conv2d(prev_channels, 1, kernel_size=4, stride=1, padding=1)
        self.use_sigmoid = use_sigmoid
        
        if use_sigmoid:
            self.output = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor, dapi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, 3, H, W)
            dapi: 条件图像 (B, 3, H, W)
        Returns:
            预测结果 (B, 1, H', W')
        """
        if dapi is not None:
            x = torch.cat([x, dapi], dim=1)
        
        x = self.input_relu(self.input_conv(x))
        x = self.middle(x)
        x = self.output_conv(x)
        
        if self.use_sigmoid:
            x = self.output(x)
        
        return x


class Pix2PixGAN(nn.Module):
    """
    Pix2Pix GAN 封装
    
    包含：
    - Generator: U-Net 生成器
    - Discriminator: Patch 判别器
    - 损失函数
    """
    def __init__(
        self,
        input_channels: int = 3,
        cond_channels: int = 3,
        output_channels: int = 3,
        base_filters: int = 64,
        gan_mode: str = 'lsgan',
    ):
        super().__init__()
        
        self.generator = UNetGenerator(
            input_channels=input_channels,
            cond_channels=cond_channels,
            output_channels=output_channels,
            base_filters=base_filters,
        )
        
        self.discriminator = PatchDiscriminator(
            input_channels=input_channels,
            cond_channels=cond_channels,
            ndf=base_filters,
        )
        
        self.gan_mode = gan_mode
    
    def forward(self, x: torch.Tensor, dapi: torch.Tensor, detach_generator: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像（实际为 DAPI，这里统一命名）
            dapi: 条件图像（也是 DAPI）
            detach_generator: 是否分离生成器用于判别器训练
        Returns:
            (fake_image, discriminator_output)
        """
        fake_image = self.generator(x, dapi)
        
        if detach_generator:
            fake_for_disc = fake_image.detach()
        else:
            fake_for_disc = fake_image
        
        disc_output = self.discriminator(fake_for_disc, dapi)
        
        return fake_image, disc_output


def build_pix2pix_model(
    input_channels: int = 3,
    cond_channels: int = 3,
    output_channels: int = 3,
    base_filters: int = 64,
) -> Pix2PixGAN:
    """构建 Pix2Pix GAN 模型"""
    return Pix2PixGAN(
        input_channels=input_channels,
        cond_channels=cond_channels,
        output_channels=output_channels,
        base_filters=base_filters,
    )
