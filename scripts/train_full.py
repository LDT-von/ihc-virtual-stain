"""
用原始完整数据集重新训练 Flow Matching 模型 (可指定 marker)

Usage:
    python scripts/train_full.py --marker CD68
"""
import argparse
import sys
import time
import random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DAPItoIHCDataset
from src.models.flow_matching import FlowMatching, FlowMatchingConfig, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--marker', default='HLA-DR',
                   choices=['HLA-DR', 'CD68', 'CD45RO', 'Vimentin'])
    p.add_argument('--data-root', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--resume-from', default=None,
                   help='Path to latest.pt to resume from (sets out_dir to same dir).')
    return p.parse_args()


def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('epochs, batch size and learning rate must be positive')
    print('Legacy full-data FM training: no held-out score or best-model selection. '
          'Use scripts/run_final.py for ROI-disjoint development.', flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    random.seed(2026)
    np.random.seed(2026)

    data_root = args.data_root
    marker = args.marker

    train_ds = DAPItoIHCDataset(
        root=data_root, marker=marker, split="train",
        patch_size=256, augment=True,
    )
    print(f"train: {len(train_ds)} samples")
    if not train_ds:
        raise ValueError('No paired training images')
    if hasattr(train_ds.transform, 'set_random_seed'):
        train_ds.transform.set_random_seed(2026)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=False, pin_memory=device.type == 'cuda')

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
        "defaults": {"marker": marker, "seed": 2026},
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
        ck = torch.load(resume_path, map_location='cpu', weights_only=False)
        if ck['cfg']['defaults']['marker'] != marker or ck['cfg']['model'] != cfg['model']:
            raise ValueError('Resume marker/model mismatch')
        if 'optimizer' not in ck:
            raise ValueError('This old weights-only checkpoint cannot resume optimizer state; use epochN.pt')
        if 'train_args' in ck and ck['train_args'] != {'batch_size': args.batch_size, 'lr': args.lr}:
            raise ValueError('Resume must preserve batch size and learning rate')
        model.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = int(ck.get('epoch', -1)) + 1
        global_step = ck.get('global_step', start_epoch * len(train_loader))
        if 'rng' in ck:
            torch.set_rng_state(ck['rng']['torch'])
            random.setstate(ck['rng']['python'])
            np.random.set_state(ck['rng']['numpy'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(ck['rng']['cuda'])
    if start_epoch >= epochs:
        raise ValueError('Checkpoint already reached the requested total epochs')

    for epoch in range(start_epoch, epochs):
        # Epoch reseeding also supports Albumentations versions with private RNGs.
        random.seed(42+epoch)
        np.random.seed(42+epoch)
        if hasattr(train_ds.transform, 'set_random_seed'):
            train_ds.transform.set_random_seed(42+epoch)
        model.train()
        t0 = time.time()
        epoch_loss, n_b = 0.0, 0

        for batch in train_loader:
            dapi = batch["dapi"].to(device)
            ihc = batch["ihc"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = fm.forward_train(ihc, dapi)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite FM training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
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

        ckpt = {
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "cfg": cfg,
                "global_step": global_step,
                "train_args": {'batch_size': args.batch_size, 'lr': args.lr},
                "rng": {'torch': torch.get_rng_state(), 'python': random.getstate(),
                        'numpy': np.random.get_state(),
                        'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []},
        }
        torch.save(ckpt, out_dir / 'latest.tmp')
        (out_dir / 'latest.tmp').replace(out_dir / 'latest.pt')
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save(ckpt, out_dir / f"epoch{epoch}.pt")
            print(f"  [Saved] {out_dir / f'epoch{epoch}.pt'}")

    final = {"epoch": epochs - 1, "model": model.state_dict(), "cfg": cfg}
    torch.save(final, out_dir / "final.pt")
    print(f"\nDone! Output: {out_dir}")


if __name__ == "__main__":
    main()
