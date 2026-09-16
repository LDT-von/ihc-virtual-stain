"""模型构建 smoke test：CPU 上跑一次前向 + 反向，确认 graph 完整。"""
import sys
sys.path.insert(0, ".")

import torch
from src.models.flow_matching import FlowMatching, build_model, FlowMatchingConfig

print("=" * 60)
print("Flow Matching 模型 smoke test (CPU)")
print("=" * 60)

device = torch.device("cpu")
model = build_model(
    in_channels=3,
    cond_channels=3,
    base_channels=32,           # CPU smoke 用小通道
    channel_mults=(1, 2, 4),    # 浅一些的 UNet
    num_res_blocks=1,
    attention_resolutions=(8,),
    dropout=0.0,
).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

fm = FlowMatching(model, FlowMatchingConfig(num_sampling_steps=10))

# 模拟一个 batch
B, C, H, W = 2, 3, 64, 64
ihc = torch.randn(B, C, H, W)
dapi = torch.randn(B, C, H, W)

# 训练一步
loss, x_t, v_pred = fm.forward_train(ihc, dapi)
loss.backward()
print(f"Train loss: {loss.item():.4f}")

# 采样
with torch.no_grad():
    pred = fm.sample(dapi, num_steps=5)
print(f"Sample shape: {tuple(pred.shape)}")

print("=" * 60)
print("OK: 模型前向/反向/采样全跑通")
print("=" * 60)