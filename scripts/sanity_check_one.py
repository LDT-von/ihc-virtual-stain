"""Sanity check: model should at least be able to output something reasonable"""
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

fm = FlowMatching(model, FlowMatchingConfig(
    sigma_min=cfg["flow_matching"]["sigma_min"],
    method=cfg["flow_matching"]["method"],
    num_sampling_steps=cfg["flow_matching"]["num_sampling_steps"],
    solver=cfg["flow_matching"]["solver"],
))

# 读一张 val DAPI
img = np.array(Image.open(r'E:\aic\ihc-data\val\DAPI\ROI009_00_05.jpg').convert('RGB'))
print(f"DAPI: shape={img.shape} mean={img.mean():.1f} std={img.std():.1f}")
print(f"  R={img[..., 0].mean():.1f} G={img[..., 1].mean():.1f} B={img[..., 2].mean():.1f}")
real = np.array(Image.open(r'E:\aic\ihc-data\val\HLA-DR\ROI009_00_05.jpg').convert('RGB'))
print(f"REAL: shape={real.shape} mean={real.mean():.1f} std={real.std():.1f}")
print(f"  R={real[..., 0].mean():.1f} G={real[..., 1].mean():.1f} B={real[..., 2].mean():.1f}")

t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0

with torch.inference_mode():
    pred = fm.sample(t, num_steps=50)
    pred_arr = ((pred[0].clamp(-1, 1).cpu().permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)

print(f"PRED: shape={pred_arr.shape} mean={pred_arr.mean():.1f} std={pred_arr.std():.1f}")
print(f"  R={pred_arr[..., 0].mean():.1f} G={pred_arr[..., 1].mean():.1f} B={pred_arr[..., 2].mean():.1f}")

Image.fromarray(pred_arr).save(r'E:\aic\ihc-virtual-stain\results\val\HLA-DR\ROI009_00_05_sanity.jpg', quality=95)

# 比较 DAPI 和 PRED 像素差异
diff = np.abs(pred_arr.astype(np.float32) - img.astype(np.float32))
print(f"Diff PRED vs DAPI: mean={diff.mean():.1f} max={diff.max():.0f}")

diff_real = np.abs(real.astype(np.float32) - img.astype(np.float32))
print(f"Diff REAL vs DAPI: mean={diff_real.mean():.1f} max={diff_real.max():.0f}")

diff_pred_real = np.abs(pred_arr.astype(np.float32) - real.astype(np.float32))
print(f"Diff PRED vs REAL: mean={diff_pred_real.mean():.1f} max={diff_pred_real.max():.0f}")
