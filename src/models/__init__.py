"""模型模块"""
from .pix2pix_gan import Pix2PixGAN, UNetGenerator, PatchDiscriminator, build_pix2pix_model
from .resnet_unet import ResNetUNet, reconstruction_loss as resnet_reconstruction_loss
from .losses import (
    SSIMLoss, 
    L1SSIMLoss, 
    PerceptualLoss, 
    GANLoss, 
    CombinedLoss,
    CombinedLoss as ImageLoss,
)
from .flow_matching import FlowMatching, FlowMatchingConfig, build_model

__all__ = [
    # Pix2Pix GAN
    "Pix2PixGAN",
    "UNetGenerator",
    "PatchDiscriminator",
    "build_pix2pix_model",
    # ResNet UNet (encoder/decoder baseline)
    "ResNetUNet",
    "resnet_reconstruction_loss",
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
]
