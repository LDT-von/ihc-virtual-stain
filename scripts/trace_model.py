"""分析 v_pred 实际值域 - 推断模型学到了什么"""
import sys
sys.path.insert(0, r"E:\aic\ihc-virtual-stain")
import torch
import numpy as np
from PIL import Image
from src.models.flow_matching import FlowMatching, FlowMatchingConfig, build_model

ckpt = torch.load(r"E:\aic\ihc-virtual-stain\checkpoints\HLA-DR_1789646403\epoch29.pt",
                  map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
device = torch.device("cuda")

model = build_model(
    in_channels=cfg["model"]["in_channels"],
    cond_channels=cfg["model"]["cond_channels"],
    base_channels=cfg["model"]["base_channels"],
    channel_mults=tuple(cfg["model"]["channel_mults"]),
    num_res_blocks=cfg["model"]["num_res_blocks"],
    attention_resolutions=tuple(cfg["model"]["attention_resolutions"]),
    dropout=cfg["model"]["dropout"],
).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# 读一张 val DAPI
img = np.array(Image.open(r'E:\aic\ihc-data\val\DAPI\ROI009_00_05.jpg').convert('RGB'))
t_dapi = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0

with torch.inference_mode():
    # 在不同 t 上预测 v，固定 x_t 走完整的 OT 轨迹
    x0 = t_dapi
    eps = torch.randn_like(x0)
    
    print("Trace along x_t trajectory (fixed x0, eps):")
    for t_val in [1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0]:
        t_tensor = torch.full((1,), t_val, device=device)
        x_t = (1-t_val) * x0 + t_val * eps
        v = model(x_t, t_tensor, t_dapi)
        print(f"  t={t_val:.1f}: x_t mean={x_t.mean().item():+.3f} (in [-1,1]={x_t.mean().item()*127.5+127.5:.0f}), "
              f"v mean={v.mean().item():+.3f}, v std={v.std().item():.3f}")
    
    # 关键：用 model 的 v 反向积分 200 步从噪声到 x_0
    print("\nReverse ODE 200 steps:")
    x = torch.randn(1, 3, 256, 256, device=device)
    dt = 1.0 / 200
    for step in range(200):
        t_now = 1.0 - step / 200
        t_t = torch.full((1,), t_now, device=device)
        v = model(x, t_t, t_dapi)
        x = x - dt * v
        if step in [0, 50, 100, 150, 199]:
            print(f"  step={step}, t={t_now:.3f}, x mean={x.mean().item():+.3f} "
                  f"(0-255: {(x.mean().item()+1)*127.5:.0f}), v mean={v.mean().item():+.3f}")
