"""
4 Marker 批量训练脚本 - SOTA v2

支持：
- 依次训练所有 4 个 marker
- 每个 marker 使用最佳超参数
- 自动寻找/创建输出目录
- 训练完成后自动评估

用法：
    # 训练所有 marker
    python train_all_sota_v2.py
    
    # 仅训练特定 marker
    python train_all_sota_v2.py --markers HLA-DR
    
    # 从上次中断处继续
    python train_all_sota_v2.py --resume
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

MARKERS = ["HLA-DR", "CD68", "CD45RO", "Vimentin"]

# 每个 marker 的最佳超参数（可根据实验调整）
MARKER_CONFIGS = {
    "HLA-DR": {
        "epochs": 120,
        "batch_size": 8,
        "base_filters": 96,
        "lr": 2e-4,
        "lr_d": 1e-4,
        "lambda_pyramid": 50.0,
        "lambda_ssim": 10.0,
        "lambda_css": 10.0,
        "lambda_edge": 2.0,
        "ema_decay": 0.999,
    },
    "CD68": {
        "epochs": 120,
        "batch_size": 8,
        "base_filters": 96,
        "lr": 2e-4,
        "lr_d": 1e-4,
        "lambda_pyramid": 50.0,
        "lambda_ssim": 10.0,
        "lambda_css": 10.0,
        "lambda_edge": 2.0,
        "ema_decay": 0.999,
    },
    "CD45RO": {
        "epochs": 120,
        "batch_size": 8,
        "base_filters": 96,
        "lr": 2e-4,
        "lr_d": 1e-4,
        "lambda_pyramid": 50.0,
        "lambda_ssim": 10.0,
        "lambda_css": 10.0,
        "lambda_edge": 2.0,
        "ema_decay": 0.999,
    },
    "Vimentin": {
        "epochs": 120,
        "batch_size": 8,
        "base_filters": 96,
        "lr": 2e-4,
        "lr_d": 1e-4,
        "lambda_pyramid": 50.0,
        "lambda_ssim": 10.0,
        "lambda_css": 10.0,
        "lambda_edge": 2.0,
        "ema_decay": 0.999,
    },
}


def train_marker(marker: str, config: dict, resume: bool = False) -> dict:
    """训练单个 marker"""
    print("\n" + "=" * 70)
    print(f"[{marker}] 开始训练")
    print(f"[{marker}] 配置: {config}")
    print("=" * 70)
    
    output_dir = ROOT / "checkpoints" / f"sota_v2_{marker}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已有训练结果
    best_ema = output_dir / "best_ema.pt"
    best_regular = output_dir / "best.pt"
    
    if best_ema.exists():
        print(f"[{marker}] 发现已有 best_ema.pt，跳过训练")
        return {"skipped": True, "marker": marker}
    if best_regular.exists() and not resume:
        print(f"[{marker}] 发现已有 best.pt，跳过训练 (使用 --resume 强制重训)")
        return {"skipped": True, "marker": marker}
    
    # 构建训练命令
    cmd = [
        sys.executable, "train_sota_v2.py",
        "--marker", marker,
        "--epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--base_filters", str(config["base_filters"]),
        "--lr", str(config["lr"]),
        "--lr_d", str(config["lr_d"]),
        "--lambda_pyramid", str(config["lambda_pyramid"]),
        "--lambda_ssim", str(config["lambda_ssim"]),
        "--lambda_css", str(config["lambda_css"]),
        "--lambda_edge", str(config["lambda_edge"]),
        "--ema_decay", str(config["ema_decay"]),
        "--output_dir", str(output_dir),
        "--log_every", "50",
        "--save_every", "5",
    ]
    
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    
    success = result.returncode == 0
    
    # 读取训练结果
    best_ssim = None
    log_file = output_dir / "train_log.jsonl"
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()
            if lines:
                last = json.loads(lines[-1])
                best_ssim = last.get("val_ssim")
    
    return {
        "marker": marker,
        "success": success,
        "elapsed": elapsed,
        "best_ssim": best_ssim,
    }


def evaluate_marker(marker: str) -> dict:
    """评估单个 marker"""
    print(f"\n[{marker}] 评估中...")
    
    ckpt_dir = ROOT / "checkpoints" / f"sota_v2_{marker}"
    
    cmd = [
        sys.executable, "infer_sota_v2.py",
        "--marker", marker,
        "--ckpt_dir", str(ckpt_dir),
        "--split", "val",
        "--use_tta",
    ]
    
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    
    # 解析输出
    import json as json_module
    results_file = Path("./submissions") / marker / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json_module.load(f)
    
    return {}


def main():
    import json
    
    p = argparse.ArgumentParser(description="4 Marker 批量训练")
    p.add_argument("--markers", nargs="+", default=MARKERS,
                   choices=MARKERS + ["all"],
                   help="要训练的 marker")
    p.add_argument("--skip_eval", action="store_true",
                   help="跳过评估阶段")
    p.add_argument("--resume", action="store_true",
                   help="强制重新训练已有模型")
    p.add_argument("--max_parallel", type=int, default=1,
                   help="最大并行训练数")
    args = p.parse_args()
    
    if "all" in args.markers:
        markers_to_train = MARKERS
    else:
        markers_to_train = [m for m in args.markers if m in MARKERS
    
    print("=" * 70)
    print(f"[Batch Training] markers: {markers_to_train}")
    print(f"[Batch Training] resume: {args.resume}")
    print("=" * 70)
    
    results = {}
    total_t0 = time.time()
    
    for i, marker in enumerate(markers_to_train):
        print(f"\n[{i+1}/{len(markers_to_train)}] 训练 {marker}")
        
        # 训练
        train_result = train_marker(marker, MARKER_CONFIGS[marker], args.resume)
        results[marker] = {"train": train_result}
        
        if not train_result.get("skipped"):
            # 评估
            if not args.skip_eval:
                eval_result = evaluate_marker(marker)
                results[marker]["eval"] = eval_result
        
        # 打印当前进度
        print("\n" + "-" * 40)
        print(f"当前进度: {i+1}/{len(markers_to_train)}")
        for m, r in results.items():
            status = "完成" if r.get("train", {}).get("success") else ("跳过" if r.get("train", {}).get("skipped") else "失败")
            ssim = r.get("eval", {}).get("ssim_mean") or r.get("train", {}).get("best_ssim")
            ssim_str = f"SSIM={ssim:.4f}" if ssim else "N/A"
            print(f"  {m}: {status} ({ssim_str})")
        print("-" * 40)
    
    total_elapsed = time.time() - total_t0
    
    # 汇总
    print("\n" + "=" * 70)
    print("训练汇总")
    print("=" * 70)
    
    total_ssim = 0.0
    count = 0
    
    for marker in markers_to_train:
        r = results.get(marker, {})
        train_r = r.get("train", {})
        eval_r = r.get("eval", {})
        
        status = "✅" if train_r.get("success") else ("⏭️" if train_r.get("skipped") else "❌")
        
        ssim = eval_r.get("ssim_mean") or train_r.get("best_ssim")
        if ssim:
            total_ssim += ssim
            count += 1
        
        ssim_str = f"SSIM={ssim:.4f}" if ssim else "SSIM=N/A"
        psnr = eval_r.get("psnr_mean")
        psnr_str = f"PSNR={psnr:.2f}" if psnr else ""
        
        print(f"{status} {marker}: {ssim_str} {psnr_str}")
    
    if count > 0:
        avg_ssim = total_ssim / count
        print(f"\n平均 SSIM: {avg_ssim:.4f}")
        print(f"预估评测分数: {avg_ssim * 100:.2f}")
    
    print(f"\n总耗时: {total_elapsed/3600:.1f} 小时")
    print(f"输出目录: {ROOT / 'checkpoints'}")
    print("=" * 70)
    
    # 保存结果
    summary_file = ROOT / "checkpoints" / "batch_training_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存: {summary_file}")


if __name__ == "__main__":
    import json
    main()
