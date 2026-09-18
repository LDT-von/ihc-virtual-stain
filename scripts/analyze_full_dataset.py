"""统计完整版 HLA-DR 真实内容比例"""
from PIL import Image
import numpy as np
from pathlib import Path

full_hla = Path(r'E:\aic\初赛数据集（包含训练集和测试集输入）\train\HLA-DR')
files = sorted(full_hla.glob('*.jpg'))

total = len(files)
has_structure = 0
all_means = []
all_stds = []
for f in files:
    arr = np.array(Image.open(f).convert('L'))
    all_means.append(arr.mean())
    all_stds.append(arr.std())
    if arr.std() > 30:
        has_structure += 1

print(f"Total: {total}")
print(f"With structure (std>30): {has_structure} ({100*has_structure/total:.1f}%)")
print(f"Mean std: {np.mean(all_stds):.2f}")
print(f"Mean pixel: {np.mean(all_means):.2f}")

# 看结构图长啥样
struct_files = [f for f in files if np.array(Image.open(f).convert('L')).std() > 30]
print(f"\nSample structured file: {struct_files[0].name}")
arr = np.array(Image.open(struct_files[0]).convert('RGB'))
print(f"  shape={arr.shape}, mean={arr.mean():.1f}, R={arr[...,0].mean():.1f} G={arr[...,1].mean():.1f} B={arr[...,2].mean():.1f}")
print(f"  R==G==B: {np.allclose(arr[...,0], arr[...,1]) and np.allclose(arr[...,1], arr[...,2])}")

# 看完整版 DAPI
full_dapi = Path(r'E:\aic\初赛数据集（包含训练集和测试集输入）\train\DAPI')
dapi_files = sorted(full_dapi.glob('*.jpg'))
print(f"\nDAPI total: {len(dapi_files)}")
dapi_stds = [np.array(Image.open(f).convert('L')).std() for f in dapi_files[:100]]
print(f"DAPI std mean: {np.mean(dapi_stds):.2f}")

# 对比：如果都用完整版训练，能有多少真图
subset_hla = Path(r'E:\aic\ihc-data\train\HLA-DR')
subset_files = {f.name for f in subset_hla.glob('*.jpg')}
full_files = {f.name for f in full_hla.glob('*.jpg')}
missing = full_files - subset_files

missing_stds = []
for f in missing:
    arr = np.array(Image.open(full_hla / f).convert('L'))
    missing_stds.append(arr.std())
print(f"\nMissing files stats: {len(missing)}, mean std: {np.mean(missing_stds):.2f}")
print(f"Missing with structure (std>30): {sum(1 for s in missing_stds if s > 30)}")
