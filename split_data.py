"""快速验证流程：划分训练集 -> 快速训练 -> 验证 -> 测试集推理

用法：
    1. 先划分数据（只要运行一次）：
       python split_data.py --marker HLA-DR --train_ratio 0.8

    2. 快速训练 + 验证：
       python quick_train.py --marker HLA-DR --epochs 5 --batch_size 16

    3. 验证集评估效果：
       python validate.py --marker HLA-DR

    4. 效果满意后，跑测试集 + 打包提交：
       python infer_and_submit.py --marker HLA-DR
"""
import argparse
import shutil
import random
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="划分训练数据为训练集1和验证集1")
    p.add_argument("--marker", type=str, default="HLA-DR",
                   help="marker 名称: HLA-DR, CD68, CD45RO, Vimentin")
    p.add_argument("--train_ratio", type=float, default=0.8,
                   help="训练集比例，默认 80%% 训练，20%% 验证")
    p.add_argument("--data_root", type=str,
                   default="E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）")
    p.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    return p.parse_args()

def main():
    args = parse_args()
    rng = random.Random(args.seed)
    
    data_root = Path(args.data_root)
    train_dapi = data_root / "train" / "DAPI"
    train_ihc = data_root / "train" / f"IHC_{args.marker}"
    
    if not train_ihc.exists():
        train_ihc = data_root / "train" / args.marker
    
    # 验证目录存在
    val_dir = data_root / "val"
    val_dapi = val_dir / "DAPI"
    val_ihc = val_dir / f"IHC_{args.marker}"
    
    print(f"[Split] marker={args.marker}")
    print(f"[Split] 源目录: {train_dapi.parent}")
    print(f"[Split] 训练集比例: {args.train_ratio:.0%}")
    
    # 获取所有 DAPI 文件
    dapi_files = sorted([f for f in train_dapi.iterdir() if f.suffix.lower() in {'.jpg', '.png', '.jpeg', '.tif', '.tiff'}])
    total = len(dapi_files)
    print(f"[Split] 总共 {total} 张图片")
    
    # 打乱
    indices = list(range(total))
    rng.shuffle(indices)
    
    split_idx = int(total * args.train_ratio)
    train_indices = set(indices[:split_idx])
    val_indices = set(indices[split_idx:])
    
    print(f"[Split] 训练集: {len(train_indices)} 张")
    print(f"[Split] 验证集: {len(val_indices)} 张")
    
    # 创建验证集目录
    val_dapi.mkdir(parents=True, exist_ok=True)
    val_ihc.mkdir(parents=True, exist_ok=True)
    
    # 移动文件到验证集（复制，不移动，防止数据丢失）
    moved = 0
    for idx in val_indices:
        dapi_file = dapi_files[idx]
        ihc_file = train_ihc / dapi_file.name
        
        # 复制到 val 目录
        shutil.copy2(dapi_file, val_dapi / dapi_file.name)
        if ihc_file.exists():
            shutil.copy2(ihc_file, val_ihc / ihc_file.name)
        
        moved += 1
        if moved % 500 == 0:
            print(f"[Split] 已处理 {moved}/{len(val_indices)}...")
    
    print(f"[OK] 验证集划分完成！")
    print(f"[OK] 验证集 DAPI: {val_dapi}")
    print(f"[OK] 验证集 IHC:  {val_ihc}")
    print(f"\n训练集保持在原位: {train_dapi.parent}")
    print(f"\n现在开始快速训练:")
    print(f"  python quick_train.py --marker {args.marker} --epochs 5 --batch_size 16")

if __name__ == "__main__":
    main()
