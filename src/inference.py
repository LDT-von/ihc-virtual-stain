"""推理入口：加载模型 + Flow Matching 采样

用法：
    python -m src.inference --ckpt checkpoints/xxx/final.pt --marker HLA-DR
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import yaml

from .data.dataset import DAPItoIHCDataset
from .metrics.ssim_psnr import MetricAggregator, to_uint8
from .models.flow_matching import FlowMatching, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--marker", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--save_images", action="store_true")
    return p.parse_args()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = to_uint8(t)
    return Image.fromarray(arr)


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg")
    if cfg is None:
        raise ValueError("checkpoint 中未包含 cfg，无法推断参数")

    marker = args.marker or cfg["defaults"]["marker"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] ckpt={ckpt_path.name} marker={marker} device={device}")

    # 模型
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

    fm = FlowMatching(model)

    # 数据
    ds = DAPItoIHCDataset(
        root=cfg["data"]["root"],
        marker=marker,
        split=args.split,
        patch_size=cfg["data"]["patch_size"],
        augment=False,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)

    out_dir = Path(args.output_dir or f"./submissions/{ckpt_path.stem}_{int(time.time())}")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    # 推理 + 评测
    metric = MetricAggregator()
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            dapi = batch["dapi"].to(device)
            ihc_gt = batch["ihc"].to(device)
            names = batch["name"]

            steps = args.num_steps or cfg["flow_matching"]["num_sampling_steps"]
            pred = fm.sample(dapi, num_steps=steps)

            for i in range(pred.shape[0]):
                metric.update(pred[i].cpu(), ihc_gt[i].cpu())
                if args.save_images:
                    tensor_to_pil(pred[i].cpu()).save(out_dir / "images" / f"{names[i]}.png")

    result = metric.result()
    elapsed = time.time() - t0
    msg = (
        f"[Infer] marker={marker} n={metric.n} "
        f"SSIM={result['ssim']:.4f} PSNR={result['psnr']:.2f} "
        f"time={elapsed:.1f}s"
    )
    print(msg)
    (out_dir / "metrics.txt").write_text(msg + "\n", encoding="utf-8")
    print(f"[Infer] 输出 -> {out_dir}")


if __name__ == "__main__":
    main()