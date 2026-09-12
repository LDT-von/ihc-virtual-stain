"""SSIM / PSNR 评测指标。"""
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as _ssim


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """将范围 [-1, 1] 的 CHW 张量转换为 RGB uint8 图像。"""
    image = (tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5
    return image.permute(1, 2, 0).numpy().astype(np.uint8)


def ssim(prediction: torch.Tensor, target: torch.Tensor, data_range: int = 255) -> float:
    """SSIM（使用 skimage，输入必须是单张 HxWxC uint8）"""
    return float(_ssim(to_uint8(prediction), to_uint8(target), channel_axis=-1, data_range=data_range))


def ssim_gpu_single(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """单张图的 SSIM (GPU)。输入 3xHxW, 范围 [-1, 1]。
    使用全图均值方差近似（和 v6 训练时一致）。
    data_range = 2.0（因为值域是 [-1, 1]）
    """
    pred = prediction.clamp(-1, 1).unsqueeze(0)  # 1x3xHxW
    tgt = target.clamp(-1, 1).unsqueeze(0)

    C1 = (0.01 * 2) ** 2
    C2 = (0.03 * 2) ** 2

    mu1 = pred.mean(dim=[2, 3])
    mu2 = tgt.mean(dim=[2, 3])
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = ((pred - mu1[:, :, None, None]) ** 2).mean(dim=[2, 3])
    sigma2_sq = ((tgt - mu2[:, :, None, None]) ** 2).mean(dim=[2, 3])
    sigma12 = ((pred - mu1[:, :, None, None]) * (tgt - mu2[:, :, None, None])).mean(dim=[2, 3])

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def psnr_gpu_single(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """单张图的 PSNR (GPU)。data_range=2.0（值域 [-1, 1]）"""
    pred = prediction.clamp(-1, 1)
    tgt = target.clamp(-1, 1)
    mse = ((pred - tgt) ** 2).mean().item()
    if mse <= 1e-10:
        return 100.0
    return 10.0 * np.log10(4.0 / mse)


class MetricAggregator:
    """累积样本并计算平均 SSIM 与 PSNR。
    逐张计算，避免 skimage OOM。
    """

    def __init__(self) -> None:
        self.ssim_sum = 0.0
        self.psnr_sum = 0.0
        self.n = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        # 4D NCHW 批量输入：逐张算
        if prediction.dim() != 4:
            raise ValueError("MetricAggregator 需要 4D NCHW 批量输入")

        for i in range(prediction.shape[0]):
            p = prediction[i]
            t = target[i]
            self.ssim_sum += ssim_gpu_single(p, t)
            self.psnr_sum += psnr_gpu_single(p, t)
            self.n += 1

    def result(self) -> dict[str, float]:
        if self.n == 0:
            return {"ssim": 0.0, "psnr": 0.0}
        return {"ssim": self.ssim_sum / self.n, "psnr": self.psnr_sum / self.n}
