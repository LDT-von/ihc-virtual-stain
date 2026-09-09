"""SSIM / PSNR 评测指标。"""
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """将范围 [-1, 1] 的 CHW 张量转换为 RGB uint8 图像。"""
    image = (tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5
    return image.permute(1, 2, 0).numpy().astype(np.uint8)


def ssim(prediction: torch.Tensor, target: torch.Tensor, data_range: int = 255) -> float:
    return float(_ssim(to_uint8(prediction), to_uint8(target), channel_axis=-1, data_range=data_range))


def psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: int = 255) -> float:
    return float(_psnr(to_uint8(target), to_uint8(prediction), data_range=data_range))


class MetricAggregator:
    """累积样本并计算平均 SSIM 与 PSNR。"""

    def __init__(self) -> None:
        self.ssim_sum = 0.0
        self.psnr_sum = 0.0
        self.n = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        self.ssim_sum += ssim(prediction, target)
        self.psnr_sum += psnr(prediction, target)
        self.n += 1

    def result(self) -> dict[str, float]:
        if self.n == 0:
            return {"ssim": 0.0, "psnr": 0.0}
        return {"ssim": self.ssim_sum / self.n, "psnr": self.psnr_sum / self.n}
