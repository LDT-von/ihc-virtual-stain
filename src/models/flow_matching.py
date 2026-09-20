"""Flow Matching 训练与采样

训练路径 x_t=(1-t)x_0+t*epsilon，网络学习 cleanward 方向 x_0-epsilon。
采样 t 从 1 降到 0，每步增加正的 cleanward 增量。
optimal_transport 是历史配置名：当前实现只是独立噪声的直线插值，未求解 OT 耦合。
independent 使用 sigma_min+(1-sigma_min)t，因此端点保留 sigma_min 噪声。
"""
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from .unet import ConditionalUNet


@dataclass
class FlowMatchingConfig:
    sigma_min: float = 1e-5
    method: str = "optimal_transport"  # "optimal_transport" | "independent"
    num_sampling_steps: int = 50
    solver: str = "euler"             # "euler" | "heun"


class FlowMatching(nn.Module):
    """
    Flow Matching 模型封装

    训练：采样 t ∈ [0,1]，构造 x_t = (1 - t) * x_0 + t * ε（OT 时用最优传输路径）
    损失：MSE(v_θ(x_t, t), v_target)，v_target = x_0 - ε 或由 OT 决定

    采样：Euler / Heun ODE 求解
    """

    def __init__(self, model: nn.Module, config: FlowMatchingConfig | None = None):
        super().__init__()
        self.model = model
        self.cfg = config or FlowMatchingConfig()
        if self.cfg.method not in ('optimal_transport', 'independent'):
            raise ValueError('Unknown flow path')
        if not 0 <= self.cfg.sigma_min < 1:
            raise ValueError('sigma_min must be in [0,1)')

    def forward_train(
        self, x0: torch.Tensor, dapi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        训练前向传播

        Args:
            x0:   真实 IHC 图像 (B, C, H, W)，已归一化到 [-1, 1]
            dapi: DAPI 条件图 (B, C, H, W)
        Returns:
            loss: MSE 损失
            x_t:  插值状态（用于日志）
            v_pred: 预测速度场
        """
        B = x0.shape[0]
        device = x0.device

        # 采样时间步 t ∈ [0, 1]
        t = torch.rand(B, device=device)

        # 采样噪声
        epsilon = torch.randn_like(x0)

        # 构建插值 x_t
        if self.cfg.method == "optimal_transport":
            # OT 路径：x_t = (1 - t) * x_0 + t * ε
            # 目标速度：v_t = x_0 - ε（从 ε 到 x_0 的直线 OT 路径）
            x_t = (1.0 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon
            v_target = x0 - epsilon
        else:
            sigma = self.cfg.sigma_min + t * (1.0 - self.cfg.sigma_min)
            x_t = (1.0 - sigma[:, None, None, None]) * x0 + sigma[:, None, None, None] * epsilon
            v_target = x0 - epsilon

        # 预测速度
        v_pred = self.model(x_t, t, dapi)

        # MSE 损失
        loss = torch.mean((v_pred - v_target) ** 2)
        return loss, x_t, v_pred

    @torch.no_grad()
    def sample(
        self,
        dapi: torch.Tensor,
        num_steps: Optional[int] = None,
        x_init: Optional[torch.Tensor] = None,
        solver: Optional[Literal["euler", "heun"]] = None,
    ) -> torch.Tensor:
        """
        从噪声生成 IHC 图像

        Args:
            dapi:     DAPI 条件图 (B, C, H, W)
            num_steps: 采样步数（默认用 config）
            x_init:   初始噪声 (B, C, H, W)，None 时随机采样
            solver:   求解器类型（默认用 config）
        Returns:
            x0_pred:  生成的 IHC 图像 (B, C, H, W)
        """
        num_steps = self.cfg.num_sampling_steps if num_steps is None else num_steps
        solver = self.cfg.solver if solver is None else solver
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
            raise ValueError('num_steps must be a positive integer')
        if solver not in ('euler', 'heun'):
            raise ValueError('solver must be euler or heun')
        if self.cfg.method not in ('optimal_transport', 'independent'):
            raise ValueError('Unknown flow path')
        if not 0 <= self.cfg.sigma_min < 1:
            raise ValueError('sigma_min must be in [0,1)')

        B, C, H, W = dapi.shape
        device = dapi.device

        # 初始状态：纯噪声 x_1 ~ N(0, 1)
        if x_init is not None:
            if x_init.shape != dapi.shape or x_init.device != device:
                raise ValueError('Initial noise must match the condition shape/device')
            x = x_init.clone()
        else:
            x = torch.randn_like(dapi)

        path_scale = 1.0 if self.cfg.method == 'optimal_transport' else 1.0-self.cfg.sigma_min
        dt = path_scale / num_steps

        # 训练时: x_t = (1 - t) * x_0 + t * ε, v_target = x_0 - ε
        # 采样: 从 t=1 (噪声) 反向 ODE 积分到 t=0 (x_0)
        # dx/dt = epsilon-x0 = -v_target. Integrating over decreasing t
        # therefore ADDS the learned cleanward velocity. Preserve the training
        # target so existing checkpoints can be re-evaluated without retraining.
        if solver == "heun":
            # Heun's method（二阶 Runge-Kutta）
            for i in range(num_steps):
                t_now = 1.0 - i / num_steps           # 1.0 → 1/N
                t_now_t = torch.full((B,), t_now, device=device, dtype=x.dtype)

                # Euler 预测
                v1 = self.model(x, t_now_t, dapi)
                x_mid = x + dt * v1

                # Heun 修正
                t_next = 1.0 - (i + 1) / num_steps    # (N-1)/N → 0
                t_next_t = torch.full((B,), t_next, device=device, dtype=x.dtype)
                v2 = self.model(x_mid, t_next_t, dapi)
                x = x + dt * 0.5 * (v1 + v2)

        else:
            # Euler：从噪声反向积分到 x_0
            for i in range(num_steps):
                t_now = 1.0 - i / num_steps           # 1.0 → 1/N
                t_now_t = torch.full((B,), t_now, device=device, dtype=x.dtype)
                v = self.model(x, t_now_t, dapi)
                x = x + dt * v

        return x


def build_model(
    in_channels: int = 3,
    cond_channels: int = 3,
    base_channels: int = 64,
    channel_mults: tuple = (1, 2, 4, 8),
    num_res_blocks: int = 2,
    attention_resolutions: tuple = (16, 8),
    dropout: float = 0.1,
) -> ConditionalUNet:
    """构建条件 UNet 模型"""
    return ConditionalUNet(
        in_channels=in_channels,
        cond_channels=cond_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
    )
