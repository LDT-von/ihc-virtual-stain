"""Test: does model output at t=0 match noise or something sensible?"""
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
    # Test 1: t=0 with x = x0 (zero noise) 
    t = torch.zeros(1, device=device)
    x = t_dapi  # x_0
    v = model(x, t, t_dapi)
    print(f"t=0, x=x0 (DAPI): v mean={v.mean().item():.4f} std={v.std().item():.4f}")
    
    # Test 2: t=1 with x = noise
    t = torch.ones(1, device=device)
    x = torch.randn_like(t_dapi)
    v = model(x, t, t_dapi)
    print(f"t=1, x=randn: v mean={v.mean().item():.4f} std={v.std().item():.4f}")
    
    # Test 3: t=0 with x = noise
    t = torch.zeros(1, device=device)
    x = torch.randn_like(t_dapi)
    v = model(x, t, t_dapi)
    print(f"t=0, x=randn: v mean={v.mean().item():.4f} std={v.std().item():.4f}")

    # Test 4: training flow direction - what's x at different t values?
    # Train formula: x_t = (1-t)*x0 + t*eps
    x0 = t_dapi
    eps = torch.randn_like(x0)
    
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        t_tensor = torch.full((1,), t_val, device=device)
        x_t = (1-t_val) * x0 + t_val * eps
        v = model(x_t, t_tensor, t_dapi)
        v_target = x0 - eps
        mse = ((v - v_target)**2).mean().item()
        print(f"t={t_val}: x_t mean={x_t.mean().item():.3f}, v_pred mean={v.mean().item():.3f}, v_target mean={v_target.mean().item():.3f}, MSE={mse:.4f}")
    
    # Critical test: what does the model predict at t=0 for x_t = x0?
    # At t=0, x_t = x0, v_target = x0 - eps
    # If model predicts v ≈ x0 - eps, then sampling should work
    print("\n--- SAMPLE with num_steps=50 ---")
    fm = FlowMatching(model, FlowMatchingConfig(
        sigma_min=cfg["flow_matching"]["sigma_min"],
        method=cfg["flow_matching"]["method"],
        num_sampling_steps=50,
        solver="euler",
    ))
    pred = fm.sample(t_dapi, num_steps=50)
    pred_arr = ((pred[0].clamp(-1,1).cpu().permute(1,2,0).numpy()+1)*127.5).astype(np.uint8)
    print(f"Sample mean={pred_arr.mean():.1f} std={pred_arr.std():.1f}")
    
    # Try with more steps
    print("\n--- SAMPLE with num_steps=200 ---")
    pred2 = fm.sample(t_dapi, num_steps=200)
    pred_arr2 = ((pred2[0].clamp(-1,1).cpu().permute(1,2,0).numpy()+1)*127.5).astype(np.uint8)
    print(f"Sample200 mean={pred_arr2.mean():.1f} std={pred_arr2.std():.1f}")
    Image.fromarray(pred_arr2).save(r'E:\aic\ihc-virtual-stain\results\val\HLA-DR\ROI009_00_05_sanity_200steps.jpg', quality=95)
