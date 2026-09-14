"""
按 ROI 分组划分训练/验证集

核心问题：当前 split_data.py 使用随机划分，同一 ROI 的 patch 可能同时出现在
训练集和验证集中，导致 SSIM 估计虚高（数据泄露）。

解决方案：
- 训练数据来自 25 个 ROI（ROI000-ROI024）
- 测试数据来自 5 个不同的 ROI（ROI025-ROI029）
- 按 ROI 编号分组划分：按编号均匀选取 3 个 ROI 作为验证集

创建标准子目录结构：train_subset/train/DAPI, train_subset/train/IHC_HLA-DR
以与现有 DAPItoIHCDataset 兼容。

用法：
    # 划分所有 marker
    python split_by_roi.py --marker all --force

    # 训练（使用 train_subset）
    python train_diffvs_latent_fm.py --marker HLA-DR --use_val_split --epochs 20

    # 验证（使用 val/）
    python train_diffvs_latent_fm.py --marker HLA-DR --epochs 1  # 不加 --use_val_split
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args():
    p = argparse.ArgumentParser(description="按 ROI 分组划分训练/验证集")
    p.add_argument("--marker", type=str, default='all',
                   help="marker 名称: HLA-DR, CD68, CD45RO, Vimentin, 或 'all'")
    p.add_argument("--val_rois", type=int, default=3,
                   help="验证集 ROI 数量（默认 3，即 3/25 ≈ 12%%）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--data_root", type=str,
                   default="E:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）")
    p.add_argument("--dry_run", action="store_true", help="仅打印划分方案，不实际复制文件")
    p.add_argument("--force", action="store_true", help="强制重新划分（清理旧目录）")
    return p.parse_args()


def extract_roi_id(filename: str) -> Optional[str]:
    """从文件名提取 ROI 编号，如 'ROI000_00_00.jpg' -> 'ROI000'"""
    parts = filename.split("_")
    if parts and parts[0].startswith("ROI"):
        return parts[0]
    return None


def main():
    args = parse_args()
    import random
    rng = random.Random(args.seed)

    data_root = Path(args.data_root)
    train_dapi = data_root / "train" / "DAPI"

    if not train_dapi.exists():
        print(f"错误：训练目录不存在 {train_dapi}")
        return

    print("=" * 60)
    print(f"[ROI Split] marker={args.marker} val_rois={args.val_rois}")
    print("=" * 60)

    # Step 1: 收集所有 ROI
    dapi_files = sorted([f for f in train_dapi.iterdir() if f.suffix.lower() in SUPPORTED_EXTS])
    roi_to_files = defaultdict(list)
    for f in dapi_files:
        roi_id = extract_roi_id(f.name)
        if roi_id:
            roi_to_files[roi_id].append(f)

    all_rois = sorted(roi_to_files.keys())
    print(f"[Data] 共发现 {len(all_rois)} 个 ROI")
    for roi, files in sorted(roi_to_files.items()):
        print(f"  {roi}: {len(files)} 张")

    # Step 2: 按 ROI 分组划分
    num_val = min(args.val_rois, len(all_rois) - 1)
    rng.shuffle(all_rois)
    val_rois = set(all_rois[:num_val])
    train_rois = set(all_rois[num_val:])

    print(f"\n[Split] 验证集 ROI ({len(val_rois)}): {sorted(val_rois)}")
    print(f"[Split] 训练集 ROI ({len(train_rois)}): {sorted(train_rois)}")
    val_file_count = sum(len(roi_to_files[r]) for r in val_rois)
    train_file_count = sum(len(roi_to_files[r]) for r in train_rois)
    print(f"[Split] 验证集: {val_file_count} 张")
    print(f"[Split] 训练集: {train_file_count} 张")

    if args.dry_run:
        print("\n[DRY RUN] 未实际复制文件")
        return

    # 清理旧目录（如果指定 --force）
    if args.force:
        for sub in ["val", "train_subset"]:
            sub_path = data_root / sub
            if sub_path.exists():
                print(f"[Cleanup] 删除 {sub_path}")
                shutil.rmtree(sub_path)

    # Step 3: 为每个 marker 划分
    markers = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin'] if args.marker == 'all' else [args.marker]

    # 创建 val/ 和 train_subset/train/ 目录
    # 先从第一个 marker 初始化目录结构（目录只需创建一次）
    val_dir = data_root / "val"
    val_dapi = val_dir / "DAPI"
    train_subset_train = data_root / "train_subset" / "train"

    val_dapi.mkdir(parents=True, exist_ok=True)
    (train_subset_train / "DAPI").mkdir(parents=True, exist_ok=True)

    total_val, total_train = 0, 0
    for marker in markers:
        # 目录名可能是 "IHC_X" 或直接是 "X"，尝试两种
        train_ihc = data_root / "train" / f"IHC_{marker}"
        if not train_ihc.exists():
            train_ihc = data_root / "train" / marker
        if not train_ihc.exists():
            print(f"  [{marker}] 跳过：train/{marker} 不存在")
            continue

        print(f"\n[{marker}] 处理中...")

        # val IHC 目录
        val_ihc = val_dir / marker
        val_ihc.mkdir(parents=True, exist_ok=True)

        # train_subset IHC 目录
        train_subset_ihc = train_subset_train / marker
        train_subset_ihc.mkdir(parents=True, exist_ok=True)

        # 复制验证集
        val_count = 0
        for roi_id in val_rois:
            for f in roi_to_files[roi_id]:
                shutil.copy2(f, val_dapi / f.name)
                ihc_file = train_ihc / f.name
                if ihc_file.exists():
                    shutil.copy2(ihc_file, val_ihc / f.name)
                val_count += 1

        # 复制训练子集
        train_count = 0
        for roi_id in train_rois:
            for f in roi_to_files[roi_id]:
                shutil.copy2(f, train_subset_train / "DAPI" / f.name)
                ihc_file = train_ihc / f.name
                if ihc_file.exists():
                    shutil.copy2(ihc_file, train_subset_ihc / f.name)
                train_count += 1

        print(f"  [{marker}] val={val_count}, train={train_count}")
        total_val += val_count
        total_train += train_count

    print(f"\n{'=' * 60}")
    print(f"[OK] 划分完成！")
    print(f"  验证集 (val/):         {val_dir}")
    print(f"  训练子集 (train_subset/train/): {train_subset_train}")
    print(f"  总计：val={total_val}, train={total_train}")
    print(f"\n训练命令（使用 train_subset）：")
    for m in markers:
        print(f"  python train_diffvs_latent_fm.py --marker {m} --use_val_split --epochs 20")


if __name__ == "__main__":
    main()
