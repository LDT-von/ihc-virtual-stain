"""检查原始完整数据集的 train/DAPI 是否有真实结构"""
from PIL import Image
import numpy as np
from pathlib import Path

dapi_dir = Path(r'E:\aic\初赛数据集（包含训练集和测试集输入）\train\DAPI')
files = sorted(dapi_dir.glob('*.jpg'))[:200]

stds = []
means = []
for f in files:
    arr = np.array(Image.open(f).convert('L'))
    stds.append(arr.std())
    means.append(arr.mean())

print(f"Original train/DAPI (first 200):")
print(f"  Mean std: {np.mean(stds):.2f}")
print(f"  Mean pixel: {np.mean(means):.2f}")
print(f"  Std>30: {sum(1 for s in stds if s > 30)} / 200")

# 检查 test/DAPI
test_dapi = Path(r'E:\aic\初赛数据集（包含训练集和测试集输入）\test\DAPI')
test_files = sorted(test_dapi.glob('*.jpg'))[:20]
test_stds = []
for f in test_files:
    arr = np.array(Image.open(f).convert('L'))
    test_stds.append(arr.std())
print(f"\nOriginal test/DAPI (first 20):")
print(f"  Mean std: {np.mean(test_stds):.2f}")
print(f"  Std>30: {sum(1 for s in test_stds if s > 30)} / 20")

# 读几张 test DAPI 看看
print("\nTest DAPI samples:")
for f in test_files[:3]:
    arr = np.array(Image.open(f).convert('RGB'))
    print(f"  {f.name}: mean={arr.mean():.1f} R={arr[...,0].mean():.1f} G={arr[...,1].mean():.1f} B={arr[...,2].mean():.1f} "
          f"R==G==B: {np.allclose(arr[...,0], arr[...,1])}")
