"""
用原始完整数据集重新训练 Flow Matching 模型 (可指定 marker)

Usage:
    python scripts/train_full.py --marker CD68
"""
import argparse
import sys
sys.path.insert(0, r"E:\aic\ihc-virtual-stain")
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DAPItoIHCDataset
from src.models.flow_matching import FlowMatching, FlowMatchingConfig, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--marker', default='HLA-DR',
                   choices=['HLA-DR', 'CD68', 'CD45RO', 'Vimentin'])
    p.add_argument('--data-root', default=r'E:\aic\初赛数据集（包含训练集和测试集输入）')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--resume-from', default=None,
                   help='Path to latest.pt to resume from (sets out_dir to same dir).')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    data_root = args.data_root
    marker = args.marker

    train_ds = DAPItoIHCDataset(
        root=data_root, marker=marker, split="train",
        patch_size=256, augment=True,
    )
    test_ds = DAPItoIHCDataset(
        root=data_root, marker=marker, split="test",
        patch_size=256, augment=False,
    )

    print(f"train: {len(train_ds)} samples")
    print(f"test: {len(test_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    safe_marker = marker.replace('/', '_')
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            resume_path = (Path.cwd() / resume_path).resolve()
        out_dir = resume_path.parent
    else:
        out_dir = Path(f"./checkpoints/{safe_marker}_full_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"./logs/train_full_{safe_marker}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs
    global_step = 0
    start_epoch = 0
    if args.resume_from:
        print(f"[Resume] loading model+optimizer from {resume_path}")
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'], strict=True)
        if 'optimizer' in ck:
            optimizer.load_state_dict(ck['optimizer'])
        start_epoch = int(ck.get('epoch', -1)) + 1
        global_step = start_epoch * (6296 // args.batch_size)

    for epoch in range(start_epoch, epochs):
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
                msg = f"[{marker}][step {global_step}] epoch={epoch} loss={loss.item():.4f}"
                print(msg)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

        avg = epoch_loss / max(n_b, 1)
        elapsed = time.time() - t0
        msg = f"[{marker}][epoch {epoch}] avg_loss={avg:.4f} time={elapsed:.1f}s"
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            ckpt = {
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "cfg": cfg,
            }
            torch.save(ckpt, out_dir / f"epoch{epoch}.pt")
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg},
                       out_dir / "latest.pt")
            print(f"  [Saved] {out_dir / f'epoch{epoch}.pt'}")

    final = {"epoch": epochs - 1, "model": model.state_dict(), "cfg": cfg}
    torch.save(final, out_dir / "final.pt")
    print(f"\nDone! Output: {out_dir}")


if __name__ == "__main__":
    main()
