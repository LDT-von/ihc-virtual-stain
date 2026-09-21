"""用 val 数据重新训练 flow matching - 小模型 + 强增强"""
import sys
sys.path.insert(0, r"E:\aic\ihc-virtual-stain")
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import DAPItoIHCDataset
from src.models.flow_matching import FlowMatching, FlowMatchingConfig, build_model
from src.metrics.ssim_psnr import MetricAggregator


def main():
    device = torch.device("cuda")
    torch.manual_seed(2026)
    
    data_root = r"E:\aic\ihc-data3"
    marker = "HLA-DR"
    
    # 加载
    train_ds = DAPItoIHCDataset(
        root=data_root, marker=marker, split="train",
        patch_size=256, augment=True,
    )
    val_ds = DAPItoIHCDataset(
        root=data_root, marker=marker, split="val",
        patch_size=256, augment=False,
    )
    print(f"train: {len(train_ds)}, val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                               num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    
    # 小模型（避免过拟合 552 张）
    model = build_model(
        in_channels=3, cond_channels=3,
        base_channels=32,  # 默认 64，减半
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.2,  # 增大 dropout
    ).to(device)
    
    cfg = {
        "model": {"in_channels": 3, "cond_channels": 3, "base_channels": 32,
                  "channel_mults": [1, 2, 4, 8], "num_res_blocks": 2,
                  "attention_resolutions": [16, 8], "dropout": 0.2},
        "flow_matching": {"sigma_min": 1e-5, "method": "optimal_transport",
                          "num_sampling_steps": 50, "solver": "euler"},
        "data": {"patch_size": 256, "root": data_root},
        "defaults": {"marker": marker},
    }
    
    fm = FlowMatching(model, FlowMatchingConfig(
        sigma_min=1e-5, method="optimal_transport",
        num_sampling_steps=50, solver="euler",
    ))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4,  # 较大 lr
                                   weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    
    out_dir = Path(f"./checkpoints/HLA-DR_val_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"./logs/train_sd_val_552.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    epochs = 80
    best_ssim = 0.0
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        epoch_loss, n_b = 0.0, 0
        for batch in train_loader:
            dapi = batch["dapi"].to(device)
            ihc = batch["ihc"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = fm.forward_train(ihc, dapi)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
            global_step += 1
            if global_step % 10 == 0:
                msg = f"[step {global_step}] epoch={epoch} loss={loss.item():.4f}"
                print(msg)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")
        
        avg = epoch_loss / max(n_b, 1)
        elapsed = time.time() - t0
        msg = f"[epoch {epoch}] avg_loss={avg:.4f} time={elapsed:.1f}s"
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        
        # 每 5 epoch 评估 + 保存
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            metric = MetricAggregator()
            with torch.inference_mode():
                for batch in val_loader:
                    dapi = batch["dapi"].to(device)
                    real = batch["ihc"].to(device)
                    pred = fm.sample(dapi, num_steps=50)
                    for j in range(pred.shape[0]):
                        metric.update(pred[j], real[j])
            r = metric.result()
            msg = f"[Eval epoch {epoch}] SSIM={r['ssim']:.4f} PSNR={r['psnr']:.2f}"
            print(msg)
            with open(log_path, "a") as f:
                f.write(msg + "\n")
            
            # 保存 best
            if r['ssim'] > best_ssim:
                best_ssim = r['ssim']
                ckpt = {
                    "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "cfg": cfg,
                    "ssim": r['ssim'], "psnr": r['psnr'],
                }
                torch.save(ckpt, out_dir / "best.pt")
                print(f"  [Best] SSIM={best_ssim:.4f}")
            
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg},
                      out_dir / f"epoch{epoch}.pt")
    
    # 最终
    final = {"epoch": epochs-1, "model": model.state_dict(), "cfg": cfg,
             "best_ssim": best_ssim}
    torch.save(final, out_dir / "final.pt")
    print(f"\nDone. Best SSIM: {best_ssim:.4f}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
