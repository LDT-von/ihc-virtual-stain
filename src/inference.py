"""生成赛题规定格式的虚拟染色结果。

示例：
    python -m src.inference --ckpt checkpoints/HLA-DR_xxx/epoch109.pt \
      --marker HLA-DR --data-root "E:/.../初赛数据集（包含训练集和测试集输入）" \
      --split test --output-dir results

测试集输出严格写入 ``results/test/<marker>/<输入名>_fake.jpg``。
"""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
import yaml

from .data.dataset import DAPItoIHCDataset
from .metrics.ssim_psnr import MetricAggregator, to_uint8
from .models.flow_matching import FlowMatching, FlowMatchingConfig, build_model


def parse_args():
    parser = argparse.ArgumentParser(description="DAPI 到 IHC 的自动推理与结果导出")
    parser.add_argument("--ckpt", required=True, help="训练得到的 .pt checkpoint")
    parser.add_argument("--marker", default=None, help="目标标记；默认读取 checkpoint 配置")
    parser.add_argument("--data-root", default=None, help="包含 train/ 和 test/ 的数据根目录")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output-dir", default="results", help="结果根目录，默认 results")
    parser.add_argument("--num-steps", type=int, default=None, help="覆盖采样步数")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-samples", type=int, default=None, help="仅用于本地冒烟测试")
    return parser.parse_args()


def build_flow_model(cfg: dict, checkpoint: dict, device: torch.device) -> FlowMatching:
    model = build_model(
        in_channels=cfg["model"]["in_channels"],
        cond_channels=cfg["model"]["cond_channels"],
        base_channels=cfg["model"]["base_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["num_res_blocks"],
        attention_resolutions=tuple(cfg["model"]["attention_resolutions"]),
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    flow_cfg = cfg["flow_matching"]
    return FlowMatching(
        model,
        FlowMatchingConfig(
            sigma_min=flow_cfg["sigma_min"],
            method=flow_cfg["method"],
            num_sampling_steps=flow_cfg["num_sampling_steps"],
            solver=flow_cfg["solver"],
        ),
    )


def main():
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality 必须在 1 到 100 之间")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples 必须大于 0")

    checkpoint_path = Path(args.ckpt)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg")
    if cfg is None:
        raise ValueError("checkpoint 未包含 cfg，无法重建模型")

    marker = args.marker or cfg["defaults"]["marker"]
    data_root = Path(args.data_root or cfg["data"]["root"])
    dapi_dir = data_root / args.split / "DAPI"
    if not dapi_dir.is_dir():
        raise FileNotFoundError(
            f"找不到输入目录：{dapi_dir}。--data-root 必须直接包含 train/ 和 test/。"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] checkpoint={checkpoint_path.name} marker={marker} device={device}")
    dataset = DAPItoIHCDataset(
        root=data_root,
        marker=marker,
        split=args.split,
        patch_size=cfg["data"]["patch_size"],
        augment=False,
    )
    if not dataset:
        raise RuntimeError(f"输入目录中没有可识别图像：{dapi_dir}")
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    output_dir = Path(args.output_dir) / args.split / marker
    output_dir.mkdir(parents=True, exist_ok=True)
    flow = build_flow_model(cfg, checkpoint, device)
    num_steps = args.num_steps or cfg["flow_matching"]["num_sampling_steps"]
    metric = MetricAggregator() if args.split == "val" else None

    start_time = time.time()
    written = 0
    with torch.inference_mode():
        for batch in loader:
            dapi = batch["dapi"].to(device, non_blocking=True)
            prediction = flow.sample(dapi, num_steps=num_steps)
            for index, name in enumerate(batch["name"]):
                image = Image.fromarray(to_uint8(prediction[index].cpu()))
                target = output_dir / f"{name}_fake.jpg"
                image.save(target, format="JPEG", quality=args.jpeg_quality)
                written += 1
                if metric is not None:
                    metric.update(prediction[index].cpu(), batch["ihc"][index])

    expected = len(dataset)
    actual = sum(1 for path in output_dir.glob("*_fake.jpg") if path.is_file())
    if written != expected or actual != expected:
        raise RuntimeError(
            f"输出文件数量不完整：期望 {expected}，本次写入 {written}，目录中有 {actual}"
        )
    elapsed = time.time() - start_time
    print(f"[Infer] 已生成 {written}/{expected} 张 JPG：{output_dir}")
    print(f"[Infer] 文件名：<输入名>_fake.jpg；耗时 {elapsed:.1f}s")
    if metric is not None:
        result = metric.result()
        print(f"[Infer] val SSIM={result['ssim']:.4f} PSNR={result['psnr']:.2f}")


if __name__ == "__main__":
    main()
