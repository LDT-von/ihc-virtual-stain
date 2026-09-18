"""用原始完整数据集重新训练 - 然后直接生成 test 提交"""
import sys
sys.path.insert(0, r"E:\aic\ihc-virtual-stain")
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import DAPItoIHCDataset
from src.models.flow_matching import FlowMatching, FlowMatchingConfig, build_model


def main():
    device = torch.device("cuda")
    torch.manual_seed(42)
    
    # 使用原始完整数据集 (6296 张训练)
    data_root = r"E:\aic\初赛数据集（包含训练集和测试集输入）"
    marker = "HLA-DR"
    
    train_ds = DAPItoIHCDataset(
        root=data_root, marker=marker, split="train",
        patch_size=256, augment=True,
    )
    test_ds = DAPItoIHCDataset(
        root=data_root, marker=marker, split="test",
        patch_size=256, augment=False,
    )
    
    # 找有多少配对
    print(f"train: {len(train_ds)} samples")
    print(f"test: {len(test_ds)} samples")
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                               num_workers=0, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    
    # 模型 - base_channels=64 正常大小
    model = build_model(
        in_channels=3, cond_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.1,
    ).to(device)
    
    cfg = {
        "model": {"in_channels": 3, "cond_channels": 3, "base_channels": 64,
                  "channel_mults": [1, 2, 4, 8], "num_res_blocks": 2,
                  "attention_resolutions": [16, 8], "dropout": 0.1},
        "flow_matching": {"sigma_min": 1e-5, "method": "optimal_transport",
                          "num_sampling_steps": 50, "solver": "euler"},
        "data": {"patch_size": 256, "root": data_root},
        "defaults": {"marker": marker},
    }
    
    fm = FlowMatching(model, FlowMatchingConfig(
        sigma_min=1e-5, method="optimal_transport",
        num_sampling_steps=50, solver="euler",
    ))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    out_dir = Path(f"./checkpoints/HLA-DR_full_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path("./logs/train_full.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    epochs = 30
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
            if global_step % 50 == 0:
                msg = f"[step {global_step}] epoch={epoch} loss={loss.item():.4f}"
                print(msg)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
        
        avg = epoch_loss / max(n_b, 1)
        elapsed = time.time() - t0
        msg = f"[epoch {epoch}] avg_loss={avg:.4f} time={elapsed:.1f}s"
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        
        # 每 5 epoch 保存
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            ckpt = {
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "cfg": cfg,
            }
            torch.save(ckpt, out_dir / f"epoch{epoch}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg},
                      out_dir / "latest.pt")
            print(f"  [Saved] {out_dir / f'epoch{epoch}.pt'}")
    
    final = {"epoch": epochs-1, "model": model.state_dict(), "cfg": cfg}
    torch.save(final, out_dir / "final.pt")
    print(f"\nDone! Output: {out_dir}")


if __name__ == "__main__":
    main()
