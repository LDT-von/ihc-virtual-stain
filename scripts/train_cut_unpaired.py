"""
自动在 4 个 marker 上运行 CUT/FastCUT unpaired 训练

与 pix2pix_sota.py 的区别：
- pix2pix_sota: paired 训练，有 pixel-level L1 监督
- CUT/FastCUT: unpaired 训练，用 PatchNCE 对比学习约束

用法：
    # 默认：FastCUT unpaired（推荐，比标准 CUT 快）
    python scripts/train_cut_unpaired.py

    # 标准 CUT
    python scripts/train_cut_unpaired.py --CUT_mode CUT

    # 仅跑一个 marker
    python scripts/train_cut_unpaired.py --markers HLA-DR --epochs 80

    # paired 模式（用你已有的 paired 数据）
    python scripts/train_cut_unpaired.py --mode paired
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()


def train_one_cut(marker: str, mode: str, CUT_mode: str,
                  epochs: int, batch_size: int, lr: float,
                  lambda_GAN: float, lambda_NCE: float,
                  n_epochs_decay: int, nce_T: float,
                  num_workers: int, save_every: int):
    """启动一次 CUT 训练"""
    marker_dir = ROOT / "checkpoints" / f"cut_{mode}_{CUT_mode}_{marker}"
    marker_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.train_cut",
        "--marker", marker,
        "--mode", mode,
        "--CUT_mode", CUT_mode,
        "--epochs", str(epochs),
        "--n_epochs_decay", str(n_epochs_decay),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--lambda_GAN", str(lambda_GAN),
        "--lambda_NCE", str(lambda_NCE),
        "--nce_T", str(nce_T),
        "--num_patches", "256",
        "--nce_layers", "0,4,8,12,16",
        "--ngf", "64",
        "--ndf", "64",
        "--n_layers_D", "3",
        "--pool_size", "0",
        "--log_every", "50",
        "--save_every", str(save_every),
        "--val_split", "0.05",
        "--num_workers", str(num_workers),
        "--output_dir", str(marker_dir),
    ]
    
    print("\n" + "=" * 70)
    print(f"[CUT] marker={marker} mode={mode} CUT_mode={CUT_mode}")
    print(f"[CUT] epochs={epochs}+{n_epochs_decay} bs={batch_size} lr={lr}")
    print(f"[CUT] lambda_GAN={lambda_GAN} lambda_NCE={lambda_NCE} nce_T={nce_T}")
    print(f"[CUT] 输出目录: {marker_dir}")
    print("=" * 70)
    
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(description="CUT/FastCUT 自动化训练脚本")
    p.add_argument("--markers", nargs="+",
                   default=["HLA-DR", "CD68", "CD45RO", "Vimentin"],
                   help="要训练的 marker")
    p.add_argument("--mode", type=str, default="unpaired",
                   choices=["unpaired", "paired", "cut_paired"],
                   help="'unpaired': DAPI/IHC 独立采样 (推荐); 'paired': 配对采样但用 CUT Loss")
    p.add_argument("--CUT_mode", type=str, default="FastCUT",
                   choices=["CUT", "FastCUT"],
                   help="'FastCUT': 轻量单向翻译 (推荐); 'CUT': 标准 CUT")
    p.add_argument("--epochs", type=int, default=80,
                   help="训练 epoch 数 (不含 decay)")
    p.add_argument("--n_epochs_decay", type=int, default=40,
                   help="学习率线性衰减的 epoch 数")
    p.add_argument("--batch_size", type=int, default=8,
                   help="批次大小 (FastCUT 可较大)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_GAN", type=float, default=1.0,
                   help="GAN loss 权重")
    p.add_argument("--lambda_NCE", type=float, default=1.0,
                   help="PatchNCE loss 权重 (FastCUT 推荐 10.0)")
    p.add_argument("--nce_T", type=float, default=0.07,
                   help="NCE temperature")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_every", type=int, default=5)
    
    args = p.parse_args()

    # FastCUT 推荐参数（来自论文）
    if args.CUT_mode == "FastCUT" and args.lambda_NCE == 1.0:
        print("[Info] FastCUT 模式，推荐 lambda_NCE=10.0，当前设为 1.0（可调）")
    
    results = {}
    for marker in args.markers:
        ok = train_one_cut(
            marker=marker,
            mode=args.mode,
            CUT_mode=args.CUT_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_GAN=args.lambda_GAN,
            lambda_NCE=args.lambda_NCE,
            n_epochs_decay=args.n_epochs_decay,
            nce_T=args.nce_T,
            num_workers=args.num_workers,
            save_every=args.save_every,
        )
        results[marker] = "OK" if ok else "FAILED"
        if not ok:
            print(f"[FAIL] {marker} 训练失败，停止")
            break

    print("\n" + "=" * 40)
    print("CUT 训练结果汇总：")
    for marker, status in results.items():
        print(f"  {marker}: {status}")
    print("=" * 40)


if __name__ == "__main__":
    main()
