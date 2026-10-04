"""模型模块"""
from .pix2pix_gan import Pix2PixGAN, UNetGenerator, PatchDiscriminator, build_pix2pix_model
from .losses import (
    SSIMLoss,
    L1SSIMLoss,
    PerceptualLoss,
    GANLoss,
    CombinedLoss,
    CombinedLoss as ImageLoss,
)
from .flow_matching import FlowMatching, FlowMatchingConfig, build_model
from .stable_diffusion import (
    SimpleVirtualStainDiffusion,
    SimpleDiffusionConfig,
    SimpleDiffusionUNet,
    build_simple_stain_sd,
)
from .marigold_ihc import (
    MarigoldIHC,
    MarigoldIHCWrapper,
    MarigoldIHCConfig,
    build_marigold_ihc,
)
from .enhanced_unetpp import (
    EnhancedUNetPlusPlus,
    build_enhanced_unet,
    combined_loss,
    ms_ssim_loss,
    edge_loss,
)
from .transformer_ihc import (
    HybridCNFTransformer,
    AttentionUNet,
    build_hybrid_cnf_ihc,
    build_attention_unet,
)

__all__ = [
    # Pix2Pix GAN
    "Pix2PixGAN",
    "UNetGenerator",
    "PatchDiscriminator",
    "build_pix2pix_model",
    # 损失函数
    "SSIMLoss",
    "L1SSIMLoss",
    "PerceptualLoss",
    "GANLoss",
    "CombinedLoss",
    "ImageLoss",
    # Flow Matching (保留旧模型作为备份)
    "FlowMatching",
    "FlowMatchingConfig",
    "build_model",
    # Stable Diffusion (简化版 - 像素空间扩散)
    "SimpleVirtualStainDiffusion",
    "SimpleDiffusionConfig",
    "SimpleDiffusionUNet",
    "build_simple_stain_sd",
    # Marigold-style Latent Diffusion (SOTA)
    "MarigoldIHC",
    "MarigoldIHCConfig",
    "build_marigold_ihc",
    # Enhanced UNet++ (Dense connections + Multi-scale attention)
    "EnhancedUNetPlusPlus",
    "build_enhanced_unet",
    "combined_loss",
    "ms_ssim_loss",
    "edge_loss",
    # Hybrid CNN-Attention Transformer
    "HybridCNFTransformer",
    "AttentionUNet",
    "build_hybrid_cnf_ihc",
    "build_attention_unet",
]
