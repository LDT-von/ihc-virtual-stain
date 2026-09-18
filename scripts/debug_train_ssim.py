"""测试在 train 数据上 evaluate 看看 SSIM 是否正常"""
import sys
sys.path.insert(0, r"E:\aic\ihc-virtual-stain")
import torch
from torch.utils.data import DataLoader
from src.data.dataset import DAPItoIHCDataset
from src.models.flow_matching import FlowMatching, FlowMatchingConfig, build_model
from src.metrics.ssim_psnr import MetricAggregator

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

ds = DAPItoIHCDataset(
    root=r"E:\aic\ihc-data",
    marker="HLA-DR",
    split="train",
    patch_size=cfg["data"]["patch_size"],
    augment=False,
)
loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
print(f"Train size: {len(ds)}")

metric = MetricAggregator()
with torch.inference_mode():
    for i, batch in enumerate(loader):
        dapi = batch["dapi"].to(device)
        target = batch["ihc"].to(device)
        pred = fm.sample(dapi, num_steps=50)
        for j in range(pred.shape[0]):
            metric.update(pred[j], target[j])  # keep on GPU
        if i >= 50:  # 400 张够了
            break

r = metric.result()
print(f"Train subset SSIM={r['ssim']:.4f} PSNR={r['psnr']:.2f}")
