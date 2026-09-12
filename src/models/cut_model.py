"""
CUT (Contrastive Unpaired Translation) 模型

论文：Park et al., "Contrastive Learning for Unpaired Image-to-Image Translation" (ECCV 2020)
变体：FastCUT — 单向翻译，更快更稳

与 paired pix2pix 的本质区别：
- pix2pix：需要 pixel-level 配对，D(G(A),B) + L1(G(A),B)
- CUT：只需要 unpaired 数据，用 PatchNCE 对比学习约束 G(A) 的特征与 A 接近
- CUT 不需要 cycle-consistency，生成器是单向的

训练策略（支持两种模式）：
- 'paired': 用配对的 DAPI↔IHC 数据，但按 CUT unpaired 方式训练（无 pixel-level L1 监督）
- 'unpaired': 把 DAPI 和 IHC 当成两个独立域，各自独立采样
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .pix2pix_gan import UNetGenerator, PatchDiscriminator
from .patchnce import poolfeat


class NCEOptimizer:
    """
    管理 PatchNCE 的投影头 netF 的优化器
    CUT 在第一个 forward 后初始化 netF（data-dependent init）
    """
    def __init__(self, netF, lr=0.0002, beta1=0.5, beta2=0.999):
        self.netF = netF
        self.opt = torch.optim.Adam(netF.parameters(), lr=lr, betas=(beta1, beta2))
    
    def step(self):
        self.opt.step()
    
    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)
    
    def state_dict(self):
        return self.opt.state_dict()
    
    def load_state_dict(self, sd):
        self.opt.load_state_dict(sd)


class CUTModel(nn.Module):
    """
    CUT 模型
    
    核心组件：
    - G: 生成器 (A→B)
    - D: PatchGAN 判别器
    - netF: PatchNCE 投影 MLP（在 data-dependent init 时确定维度）
    - nce_layers: 提取中间特征的位置
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        n_layers_D: int = 3,
        nce_layers: str = '0,4,8,12,16',
        nce_T: float = 0.07,
        num_patches: int = 256,
        normG: str = 'instance',
        normD: str = 'instance',
        gan_mode: str = 'lsgan',
        lambda_NCE: float = 1.0,
        lambda_GAN: float = 1.0,
        lambda_idt: float = 0.0,
        pool_size: int = 0,
        flip_equivariance: bool = False,
    ):
        super().__init__()
        self.nce_T = nce_T
        self.num_patches = num_patches
        self.lambda_NCE = lambda_NCE
        self.lambda_GAN = lambda_GAN
        self.lambda_idt = lambda_idt
        self.pool_size = pool_size
        self.flip_equivariance = flip_equivariance
        self.nce_layer_list = [int(x) for x in nce_layers.split(',')]
        
        # Generator: ResNet 风格 (输出中间层特征)
        self.netG = ResNetGenerator(
            input_nc=input_channels,
            output_nc=output_channels,
            ngf=ngf,
            n_blocks=9,
            norm_type=normG,
        )
        
        # Discriminator: PatchGAN
        self.netD = NLayerDiscriminator(
            input_nc=output_channels,
            ndf=ndf,
            n_layers=n_layers_D,
            norm_type=normD,
        )
        
        # netF_dict 将在 data_dependent_initialize 时填充（每个 nce 层一个 1×1 Conv）
        self.netF_dict = nn.ModuleDict()
        self.netF = None  # 兼容旧属性
        self.nce_optimizer = None
        
        # 图像池（可选，用于增强判别器）
        self.image_pool = ImagePool(pool_size)
        
        # GAN Loss
        if gan_mode == 'lsgan':
            self.gan_criterion = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.gan_criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown gan_mode: {gan_mode}")
        
        self.real_A: torch.Tensor = None
        self.real_B: torch.Tensor = None
        self.fake_B: torch.Tensor = None
        self.loss_G: float = 0.0
        self.loss_D: float = 0.0
        self.loss_NCE: float = 0.0
        self.loss_NCE_Y: float = 0.0
    
    def data_dependent_initialize(self, real_A: torch.Tensor, real_B: torch.Tensor):
        """
        Data-dependent initialization
        在第一次 forward 后调用，确定 netF 的输入维度
        """
        # 先做一次 forward 获取特征维度
        self.eval()
        with torch.no_grad():
            _, feat_A = self.netG(real_A, layers=self.nce_layer_list, encode_only=True)
        self.train()
        
        # 创建 netF（每个 nce 层一个 1×1 Conv，自动适配任意通道数）
        # feat_A 是 dict {layer_idx: (B,C,H,W)}
        self.netF_dict = nn.ModuleDict()
        for li in self.nce_layer_list:
            if li in feat_A:
                ch = feat_A[li].shape[1]
                self.netF_dict[str(li)] = nn.Sequential(
                    nn.Conv2d(ch, 256, kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=1),
                ).to(real_A.device)
        self.nce_optimizer = NCEOptimizer(self.netF_dict)
        
        print(f"[CUT] Data-dependent init: feat_dims={{{', '.join(f'L{li}={feat_A[li].shape[1]}' for li in self.nce_layer_list if li in feat_A)}}}, nce_layers={self.nce_layer_list}")
    
    def forward_G(self, real_A: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        G(A) → fake_B，返回 fake_B 和中间特征
        """
        fake_B, feat_A = self.netG(real_A, layers=self.nce_layer_list)
        return fake_B, feat_A
    
    def forward(self, real_A: torch.Tensor, real_B: torch.Tensor,
                nce_idt: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        完整 forward
        
        Args:
            real_A: 源域 (DAPI)  (B,3,H,W)
            real_B: 目标域 (IHC)  (B,3,H,W)
            nce_idt: 是否计算 identity NCE loss（仅 FastCUT）
        Returns:
            fake_B, feat_fake, feat_real
        """
        # flip equivariance (仅训练时 FastCUT 用)
        if self.is_training() and self.flip_equivariance:
            if torch.rand(1) < 0.5:
                real_A = torch.flip(real_A, [3])
                real_B = torch.flip(real_B, [3])
                self._flipped = True
            else:
                self._flipped = False
        
        # G(A) → fake_B
        fake_B, feat_A = self.forward_G(real_A)
        self.fake_B = fake_B
        
        if nce_idt:
            # FastCUT: identity loss，G(B) → idt_B，特征应接近 B
            idt_B, feat_B = self.forward_G(real_B)
            return fake_B, feat_A, feat_B
        else:
            # 标准 CUT: 只取 G(A) 的特征
            return fake_B, feat_A, None
    
    def is_training(self) -> bool:
        return self.netG.training
    
    def compute_G_loss(self, real_A: torch.Tensor, real_B: torch.Tensor,
                       feat_fake: List[torch.Tensor],
                       nce_idt: bool = False, feat_idt: Optional[List[torch.Tensor]] = None,
                       feat_real_A: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        计算 G 的总损失：
        1. GAN loss: D(G(A)) 应判为真
        2. NCE loss: G(A) 的特征应与 A 的特征接近
        
        feat_real_A：如果外部已计算则复用，否则重新计算（内部会加 no_grad）
        """
        loss_G = 0.0
        T = self.nce_T
        
        # 1. GAN loss（无条件判别器）
        if self.lambda_GAN > 0:
            pred_fake = self.netD(self.fake_B)
            loss_G_GAN = self.gan_criterion(
                pred_fake, torch.ones_like(pred_fake)
            ) * self.lambda_GAN
            loss_G = loss_G_GAN
        
        # 2. PatchNCE loss：feat_real_A 已在 no_grad context 中计算
        if self.lambda_NCE > 0 and feat_real_A is not None:
            loss_NCE = self.compute_PatchNCE_loss(feat_fake, feat_real_A)
            loss_G = loss_G + loss_NCE * self.lambda_NCE
        elif self.lambda_NCE > 0:
            # fallback：重新计算（加 no_grad）
            with torch.no_grad():
                _, feat_real_A_fallback = self.netG(real_A, layers=self.nce_layer_list, encode_only=True)
            if isinstance(feat_real_A_fallback, dict):
                feat_real_A_fallback = [feat_real_A_fallback[li] for li in self.nce_layer_list]
            loss_NCE = self.compute_PatchNCE_loss(feat_fake, feat_real_A_fallback)
            loss_G = loss_G + loss_NCE * self.lambda_NCE
        
        # 3. Identity NCE loss (FastCUT)
        if nce_idt and feat_idt is not None and self.lambda_idt > 0:
            with torch.no_grad():
                _, feat_real_idt = self.netG(real_B, layers=self.nce_layer_list, encode_only=True)
            if isinstance(feat_real_idt, dict):
                feat_real_idt = [feat_real_idt[li] for li in self.nce_layer_list]
            loss_NCE_Y = self.compute_PatchNCE_loss(feat_idt, feat_real_idt)
            loss_G = loss_G + loss_NCE_Y * self.lambda_idt

        # 记录 loss_NCE（可能在 elif 里新定义）
        try:
            self.loss_NCE = loss_NCE.item() if isinstance(loss_NCE, torch.Tensor) else loss_NCE
        except NameError:
            self.loss_NCE = 0.0  # lambda_NCE == 0 时未定义 NCE loss
        self.loss_G = loss_G.item() if isinstance(loss_G, torch.Tensor) else loss_G

        return loss_G
    
    def compute_D_loss(self, real_B: torch.Tensor) -> torch.Tensor:
        """
        计算 D 的损失：real 应判真，fake 应判假
        CUT 的判别器是 unconditional 的，不接收 dapi 条件
        """
        # real
        pred_real = self.netD(real_B)
        loss_D_real = self.gan_criterion(
            pred_real, torch.ones_like(pred_real)
        )
        
        # fake（从图像池采样）
        fake_B_pool = self.image_pool.query(self.fake_B.detach())
        pred_fake = self.netD(fake_B_pool)
        loss_D_fake = self.gan_criterion(
            pred_fake, torch.zeros_like(pred_fake)
        )
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D = loss_D.item() if isinstance(loss_D, torch.Tensor) else loss_D
        
        return loss_D
    
    def compute_PatchNCE_loss(self, feat_q, feat_k) -> torch.Tensor:
        """
        计算多层 PatchNCE loss
        
        Args:
            feat_q: dict {layer_idx: tensor} 或 list
            feat_k: dict {layer_idx: tensor} 或 list
        """
        import math
        # 兼容 dict 和 list 两种输入格式
        if isinstance(feat_q, dict):
            feat_q = [feat_q[li] for li in self.nce_layer_list]
        if isinstance(feat_k, dict):
            feat_k = [feat_k[li] for li in self.nce_layer_list]
        
        nce_loss = 0.0
        T = self.nce_T
        
        # feat_q / feat_k 是 list，元素顺序对应 nce_layer_list
        for i, (fq, fk) in enumerate(zip(feat_q, feat_k)):
            li = self.nce_layer_list[i]
            netF = self.netF_dict[str(li)].to(fq.device)
            B, C, H, W = fq.shape
            L = H * W
            device = fq.device
            
            # 采样 num_patches 个 patch（避免显存爆炸）
            if L > self.num_patches:
                indices = torch.randint(0, L, (B, self.num_patches), device=device)
                fq_flat = fq.view(B, C, -1)   # (B, C, L)
                fk_flat = fk.view(B, C, -1)
                indices_expanded = indices.unsqueeze(1).expand(B, C, self.num_patches)
                fq_s = torch.gather(fq_flat, 2, indices_expanded)  # (B, C, P)
                fk_s = torch.gather(fk_flat, 2, indices_expanded)
                P = self.num_patches
            else:
                fq_s = fq.view(B, C, -1)
                fk_s = fk.view(B, C, -1)
                P = L
            
            # 始终 reshape 到 4D：(B, C, 1, P)，确保兼容 Conv2d
            fq_2d = fq_s.view(B, C, 1, P)  # (B, C, 1, P)
            fk_2d = fk_s.view(B, C, 1, P)
            
            # Conv2d 1×1 投影：任意 C -> 256
            fq_proj = netF(fq_2d)   # (B, C, 1, P) -> (B, 256, 1, P)
            fk_proj = netF(fk_2d)
            fq_proj = fq_proj.view(B, 256, -1)  # -> (B, 256, P)
            fk_proj = fk_proj.view(B, 256, -1)
            
            # L2 归一化 + transpose -> (B, P, 256)
            fq_proj = F.normalize(fq_proj, dim=1).transpose(1, 2)  # (B, P, 256)
            fk_proj = F.normalize(fk_proj, dim=1).transpose(1, 2)
            
            P_actual = fq_proj.shape[1]
            
            # NCE相似度: (B, P, P)
            sim = torch.bmm(fq_proj, fk_proj.transpose(-2, -1)) / T
            
            # 正样本：对角线 (B, P)
            idx = torch.arange(P_actual, device=device)
            pos_sim = sim[:, idx, idx]
            
            # 行方向 logsumexp（数值稳定）
            row_max = sim.max(dim=-1, keepdim=True).values
            sim_stable = sim - row_max
            log_sum_exp = torch.log(torch.exp(sim_stable).sum(dim=-1) + 1e-8) + row_max.squeeze(-1)
            
            # NCE loss
            nce = -pos_sim / T + log_sum_exp
            nce_loss = nce_loss + nce.mean()
        
        return nce_loss / len(feat_q)
    
    def set_requires_grad(self, nets: nn.Module, requires_grad: bool):
        """冻结/解冻网络参数"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for p in net.parameters():
                    p.requires_grad = requires_grad
    
    def train_step(self, real_A: torch.Tensor, real_B: torch.Tensor,
                   nce_idt: bool = False, optimizer_G=None, optimizer_D=None,
                   optimizer_F=None) -> dict:
        """
        一次完整的训练 step
        
        Returns:
            dict with loss values
        """
        # Forward G
        fake_B, feat_fake = self.forward_G(real_A)
        self.fake_B = fake_B

        # Update D
        if optimizer_D is not None:
            self.set_requires_grad(self.netD, True)
            optimizer_D.zero_grad()
            loss_D = self.compute_D_loss(real_B)
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0)
            optimizer_D.step()

        # Update G + netF
        if optimizer_G is not None or optimizer_F is not None:
            self.set_requires_grad(self.netD, False)
            if optimizer_G is not None:
                optimizer_G.zero_grad()
            if optimizer_F is not None:
                optimizer_F.zero_grad()
            with torch.no_grad():
                _, feat_real_A = self.netG(real_A, layers=self.nce_layer_list, encode_only=True)
                if isinstance(feat_real_A, dict):
                    feat_real_A = [feat_real_A[li] for li in self.nce_layer_list]
            loss_G = self.compute_G_loss(real_A, real_B, feat_fake,
                                        nce_idt=False, feat_idt=None,
                                        feat_real_A=feat_real_A)
            if isinstance(loss_G, torch.Tensor):
                loss_G.backward()
            if optimizer_G is not None:
                torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
                optimizer_G.step()
            if optimizer_F is not None:
                torch.nn.utils.clip_grad_norm_(self.netF_dict.parameters(), max_norm=1.0)
                optimizer_F.step()

        return {
            'loss_G': self.loss_G,
            'loss_D': self.loss_D if hasattr(self, 'loss_D') else 0.0,
            'loss_NCE': self.loss_NCE if hasattr(self, 'loss_NCE') else 0.0,
        }


# ---------------------------------------------------------------------------
# ResNet Generator (输出中间层特征) — CUT 原版风格
# ---------------------------------------------------------------------------

class ResnetBlock(nn.Module):
    def __init__(self, dim: int, norm_type: str = 'instance', use_dropout: bool = False):
        super().__init__()
        block = []
        p = 1  # padding
        # 第一个卷积
        block += [nn.ReflectionPad2d(p),
                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0),
                  self._norm(dim, norm_type)]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        # 第二个卷积
        block += [nn.ReflectionPad2d(p),
                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0),
                  self._norm(dim, norm_type)]
        self.block = nn.Sequential(*block)
    
    def _norm(self, channels: int, norm_type: str):
        if norm_type == 'instance':
            return nn.InstanceNorm2d(channels)
        elif norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        return nn.Identity()
    
    def forward(self, x):
        out = x + self.block(x)
        return out


class ResNetGenerator(nn.Module):
    """
    ResNet 风格的生成器（来自 CUT 官方实现）
    支持输出中间层特征用于 PatchNCE
    
    架构（输入 256×256）：
    - Layer 0: pad+conv+norm+relu  → 256×256, 64ch
    - Layer 1: conv+norm+relu      → 128×128, 128ch  (downsample)
    - Layer 2: conv+norm+relu      → 64×64,   256ch  (downsample)
    - Layer 3-11: 9 个 ResNet blocks → 64×64,   256ch
    - Layer 12: upconv+norm+relu   → 128×128, 128ch  (upsample)
    - Layer 13: upconv+norm+relu   → 256×256,  64ch  (upsample)
    - Layer 14-16: pad+conv+tanh   → 256×256,   3ch
    
    CUT PatchNCE 在 layer [0,4,8,12,16] 提取特征：
      0 → 256×256 (encoder output)
      4 → 64×64   (bottleneck)
      8 → 64×64   (bottleneck)
     12 → 128×128 (first upsample)
     16 → 256×256 (final output)
    
    这对应 5 个不同分辨率的特征，与 CUT 论文一致。
    """
    
    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64,
                 n_blocks: int = 9, norm_type: str = 'instance',
                 padding_type: str = 'reflect'):
        assert n_blocks >= 0
        super().__init__()
        
        if padding_type == 'reflect':
            pad3 = nn.ReflectionPad2d(3)
        elif padding_type == 'zero':
            pad3 = nn.ZeroPad2d(3)
        else:
            pad3 = nn.ReplicationPad2d(3)
        
        # Layer 0: pad+conv+norm+relu → 256×256, 64ch
        self.layer0 = nn.Sequential(
            pad3, nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            self._norm(ngf, norm_type), nn.ReLU(inplace=True)
        )
        
        # Layer 1: downsample → 128×128, 128ch
        self.layer1 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
            self._norm(ngf*2, norm_type), nn.ReLU(inplace=True)
        )
        
        # Layer 2: downsample → 64×64, 256ch
        self.layer2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
            self._norm(ngf*4, norm_type), nn.ReLU(inplace=True)
        )
        
        # Layer 3-11: ResNet blocks → 64×64, 256ch
        self.layer3 = nn.Sequential(*[
            ResnetBlock(ngf*4, norm_type) for _ in range(n_blocks)
        ])
        
        # Layer 12: upsample → 128×128, 128ch
        self.layer12 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            self._norm(ngf*2, norm_type), nn.ReLU(inplace=True)
        )
        
        # Layer 13: upsample → 256×256, 64ch
        self.layer13 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            self._norm(ngf, norm_type), nn.ReLU(inplace=True)
        )
        
        # Layer 14-16: 输出
        self.layer14 = nn.Sequential(
            pad3, nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )
        
        self.n_blocks = n_blocks
    
    def _norm(self, channels: int, norm_type: str):
        if norm_type == 'instance':
            return nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'batch':
            return nn.BatchNorm2d(channels)
        return nn.Identity()
    
    def forward(self, x, layers=None, encode_only: bool = False):
        """
        Args:
            x: (B, C, H, W)
            layers: list of layer indices to extract (e.g. [0,4,8,12,16])
            encode_only: if True, return features + final output
        """
        feat_dict = {}
        
        # Layer 0
        x = self.layer0(x)
        if 0 in layers: feat_dict[0] = x
        
        # Layer 1
        x = self.layer1(x)
        if 1 in layers: feat_dict[1] = x
        
        # Layer 2
        x = self.layer2(x)
        if 2 in layers: feat_dict[2] = x
        
        # Layer 3-11 (bottleneck)
        x = self.layer3(x)
        if 3 in layers: feat_dict[3] = x
        # CUT's layer 4 and 8 are within the ResNet blocks
        # For simplicity, we map layer 4 → end of layer3, layer 8 → also layer3
        if 4 in layers: feat_dict[4] = x
        if 8 in layers: feat_dict[8] = x
        
        # Layer 12
        x = self.layer12(x)
        if 12 in layers: feat_dict[12] = x
        
        # Layer 13
        x = self.layer13(x)
        if 13 in layers: feat_dict[13] = x
        
        # Layer 14-16
        out = self.layer14(x)
        if 16 in layers: feat_dict[16] = out
        
        if encode_only or layers:
            return out, feat_dict
        return out, {}


class NLayerDiscriminator(nn.Module):
    """
    Multi-layer PatchGAN 判别器 (CUT 原版实现)
    无条件版本：只接收图像输入，不接收 dapi 条件
    
    架构示例（n_layers=3, ndf=64）：
    - Conv2d(3→64, 4×4, s2): 128→64
    - Conv2d(64→128, 4×4, s2): 64→32  
    - Conv2d(128→256, 4×4, s2): 32→16
    - Conv2d(256→512, 4×4, s1): 16→16
    - Conv2d(512→1, 4×4, s1): 16→16
    输出：16×16 patch prediction
    """
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3,
                 norm_type: str = 'instance'):
        super().__init__()
        self.n_layers = n_layers
        
        kw = 4
        padw = 1
        nf_mult = 1
        
        # 第一个卷积：input_nc → ndf（不做 norm）
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 中间层：ndf*2 → ndf*4 → ndf*8
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(nf_mult * 2, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # 最后层：ndf*8 → ndf*8，stride 1
        nf_mult_prev = nf_mult
        nf_mult = min(nf_mult * 2, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 输出 1 通道
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.model(x)


class ImagePool:
    """图像池：存储历史 fake 图像以增强判别器"""
    def __init__(self, pool_size: int = 0):
        self.pool_size = pool_size
        self.images = []
    
    def query(self, images: torch.Tensor) -> torch.Tensor:
        """
        从池中返回图像：
        - 50% 概率返回当前 batch
        - 50% 概率从池中随机选
        """
        if self.pool_size == 0:
            return images
        
        return_images = []
        for img in images:
            img = torch.unsqueeze(img, 0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                return_images.append(img)
            else:
                if torch.rand(1) < 0.5:
                    idx = torch.randint(0, len(self.images), (1,))
                    return_images.append(self.images[idx].to(img.device))
                else:
                    self.images[torch.randint(0, len(self.images), (1,)).item()] = img
                    return_images.append(img)
        
        return torch.cat(return_images, 0)


def build_cut_model(**kwargs) -> CUTModel:
    return CUTModel(**kwargs)
