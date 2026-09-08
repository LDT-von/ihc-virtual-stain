"""训练入口

用法：
    python -m src.train --config configs/default.yaml --marker HLA-DR
    python -m src.train --data_root "E:/aic/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）"
"""
import argparse
import os
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data.dataset import DAPItoIHCDataset
from .models.flow_matching import FlowMatching, FlowMatchingConfig, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--marker", type=str, default=None, help="覆盖配置中的 marker")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--data_root", type=str, default=None, help="覆盖 data.root")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def merge_config(args, cfg):
    if args.marker:
        cfg["defaults"]["marker"] = args.marker
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    return cfg


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = merge_config(args, cfg)

    marker = cfg["defaults"]["marker"]
    seed = cfg["defaults"]["seed"]
    device = torch.device(cfg["defaults"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    print(f"[Train] marker={marker} device={device}")

    # 数据集
    train_ds = DAPItoIHCDataset(
        root=cfg["data"]["root"],
        marker=marker,
        split="train",
        patch_size=cfg["data"]["patch_size"],
        augment=cfg["data"]["augment"],
    )
    val_ds = DAPItoIHCDataset(
        root=cfg["data"]["root"],
        marker=marker,
        split="val",
        patch_size=cfg["data"]["patch_size"],
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=2
    ) if len(val_ds) > 0 else None

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

    fm = FlowMatching(model, FlowMatchingConfig(
        sigma_min=cfg["flow_matching"]["sigma_min"],
        method=cfg["flow_matching"]["method"],
        num_sampling_steps=cfg["flow_matching"]["num_sampling_steps"],
        solver=cfg["flow_matching"]["solver"],
    ))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["defaults"]["amp"] and device.type == "cuda")

    # 恢复
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Train] 恢复自 {args.resume}, epoch={start_epoch}")

    # 输出目录
    out_dir = Path(f"./checkpoints/{marker}_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("./logs") / f"train_{marker}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 训练循环
    global_step = 0
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            dapi = batch["dapi"].to(device, non_blocking=True)
            ihc = batch["ihc"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                loss, _, _ = fm.forward_train(ihc, dapi)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % cfg["train"]["log_every"] == 0:
                msg = (
                    f"[step {global_step}] epoch={epoch} "
                    f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
                )
                print(msg)
                log_path.write_text(msg + "\n", encoding="utf-8", append=True)

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        msg = f"[epoch {epoch}] avg_loss={avg_loss:.4f} time={elapsed:.1f}s"
        print(msg)
        log_path.write_text(msg + "\n", encoding="utf-8", append=True)

        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            ckpt_path = out_dir / f"epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                },
                ckpt_path,
            )
            print(f"[Train] 保存 {ckpt_path}")

    # 最终保存
    final = out_dir / "final.pt"
    torch.save({"epoch": cfg["train"]["epochs"] - 1, "model": model.state_dict(), "cfg": cfg}, final)
    print(f"[Train] 训练完成 -> {final}")


if __name__ == "__main__":
    main()