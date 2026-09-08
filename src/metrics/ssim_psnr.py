"""SSIM / PSNR 评测指标"""
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr
import numpy as np


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """[-1,1] -> uint8 [H,W,3]"""
    t = (t.clamp(-1, 1) + 1) * 127.5
    return t.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: int = 255) -> float:
    """两张 [-1,1] tensor 计算 SSIM"""
    p = to_uint8(pred)
    t = to_uint8(target)
    # channel_axis 让 SSIM 对 RGB 整体计算
    return float(_ssim(p, t, channel_axis=-1, data_range=data_range))


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: int = 255) -> float:
    p = to_uint8(pred)
    t = to_uint8(target)
    return float(_psnr(t, p, data_range=data_range))


class MetricAggregator:
    """累积一批样本的平均指标"""

    def __init__(self):
        self.ssim_sum = 0.0
        self.psnr_sum = 0.0
        self.n = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.ssim_sum += ssim(pred, target)
        self.psnr_sum += psnr(pred, target)
        self.n += 1

    def result(self) -> dict[str, float]:
        if self.n == 0:
            return {"ssim": 0.0, "psnr": 0.0}
        return {
            "ssim": self.ssim_sum / self.n,
            "psnr": self.psnr_sum / self.n,
        }

    def reset(self):
        self.ssim_sum = 0.0
        self.psnr_sum = 0.0
        self.n = 0