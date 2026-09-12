"""Pix2Pix GAN 推理脚本

用法：
    python -m src.inference_pix2pix --ckpt "checkpoints/pix2pix_HLA-DR_xxx/epoch5.pt" \
      --marker HLA-DR --data-root "E:/.../初赛数据集" \
      --split test --output-dir results
"""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data.dataset import DAPItoIHCDataset
from .metrics.ssim_psnr import MetricAggregator, to_uint8
from .models.pix2pix_gan import build_pix2pix_model


def parse_args():
    parser = argparse.ArgumentParser(description="Pix2Pix GAN 推理脚本")
    parser.add_argument("--ckpt", required=True, help="训练得到的 .pt checkpoint")
    parser.add_argument("--marker", default="HLA-DR", help="目标标记")
    parser.add_argument("--data-root",
                        default="E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）",
                        help="数据根目录")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output-dir", default="results", help="结果根目录，默认 results")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)  # 避免多进程问题
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--tta", action="store_true", help="开启 TTA（4x 翻转融合，默认开启）")
    parser.add_argument("--no-tta", action="store_true", help="禁用 TTA")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality 必须在 1 到 100 之间")
    
    checkpoint_path = Path(args.ckpt)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")
    
    print(f"[Infer] 加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    marker = args.marker
    data_root = Path(args.data_root)
    
    print(f"[Infer] marker={marker}, data_root={data_root}")
    
    # 数据集
    dapi_dir = data_root / args.split / "DAPI"
    if not dapi_dir.is_dir():
        raise FileNotFoundError(f"找不到输入目录：{dapi_dir}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Infer] device={device}")
    
    dataset = DAPItoIHCDataset(
        root=data_root,
        marker=marker,
        split=args.split,
        patch_size=256,
        augment=False,
    )
    
    if not dataset:
        raise RuntimeError(f"输入目录中没有可识别图像：{dapi_dir}")
    
    print(f"[Dataset] 找到 {len(dataset)} 张图像")
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    
    # 模型
    model = build_pix2pix_model(
        input_channels=3,
        cond_channels=3,
        output_channels=3,
        base_filters=64,
    ).to(device)
    
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    print(f"[Model] 加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 输出目录
    output_dir = Path(args.output_dir) / args.split / marker
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 指标计算
    metric = MetricAggregator() if args.split == "val" else None
    
    start_time = time.time()
    written = 0

    use_tta = args.tta and not args.no_tta
    if not use_tta:
        # 无 TTA：直接一次前向
        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader):
                dapi = batch["dapi"].to(device, non_blocking=True)
                fake_img = model.generator(dapi, dapi)
                if metric is not None:
                    metric.update(fake_img, batch["ihc"].to(device, non_blocking=True))
                for index, name in enumerate(batch["name"]):
                    image = Image.fromarray(to_uint8(fake_img[index].cpu()))
                    target = output_dir / f"{name}_fake.jpg"
                    image.save(target, format="JPEG", quality=args.jpeg_quality)
                    written += 1
                if (batch_idx + 1) % 50 == 0:
                    print(f"[Infer] 已处理 {written} 张图像...")
    else:
        # 4x TTA：原图 + HFlip + VFlip + HVFlip，取平均
        print("[Infer] TTA enabled: 4x flip averaging")
        def tta_forward(dapi_input):
            outs = [
                model.generator(dapi_input, dapi_input),
                torch.flip(model.generator(torch.flip(dapi_input, dims=(3,)), torch.flip(dapi_input, dims=(3,))), dims=(3,)),
                torch.flip(model.generator(torch.flip(dapi_input, dims=(2,)), torch.flip(dapi_input, dims=(2,))), dims=(2,)),
                torch.flip(model.generator(torch.flip(dapi_input, dims=(2, 3)), torch.flip(dapi_input, dims=(2, 3))), dims=(2, 3)),
            ]
            return torch.stack(outs).mean(dim=0)
        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader):
                dapi = batch["dapi"].to(device, non_blocking=True)
                fake_img = tta_forward(dapi)
                if metric is not None:
                    metric.update(fake_img, batch["ihc"].to(device, non_blocking=True))
                for index, name in enumerate(batch["name"]):
                    image = Image.fromarray(to_uint8(fake_img[index].cpu()))
                    target = output_dir / f"{name}_fake.jpg"
                    image.save(target, format="JPEG", quality=args.jpeg_quality)
                    written += 1
                if (batch_idx + 1) % 50 == 0:
                    print(f"[Infer] 已处理 {written} 张图像...")
    
    elapsed = time.time() - start_time
    
    print(f"[Infer] 已生成 {written}/{len(dataset)} 张 JPG：{output_dir}")
    print(f"[Infer] 耗时 {elapsed:.1f}s ({elapsed/max(written,1)*1000:.1f}ms/张), TTA={'4x flip' if use_tta else 'off'}")
    
    if metric is not None:
        result = metric.result()
        print(f"[Infer] val SSIM={result['ssim']:.4f} PSNR={result['psnr']:.2f}")


if __name__ == "__main__":
    main()
