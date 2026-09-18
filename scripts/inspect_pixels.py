"""分析 train/DAPI 和 val/HLA-DR 像素内容"""
import numpy as np
from PIL import Image

# 抽 5 张 train/DAPI
print("=== train/DAPI ===")
for name in ['ROI000_00_00.jpg', 'ROI012_07_06.jpg', 'ROI024_15_16.jpg']:
    img = np.array(Image.open(f'E:/aic/ihc-data/train/DAPI/{name}').convert('RGB'))
    print(f"{name}: shape={img.shape} mean={img.mean():.1f} "
          f"R={img[..., 0].mean():.1f} G={img[..., 1].mean():.1f} B={img[..., 2].mean():.1f} "
          f"min={img.min()} max={img.max()} std={img.std():.1f}")

print("\n=== val/HLA-DR (the supposedly real brown IHC) ===")
for name in ['ROI009_00_05.jpg', 'ROI009_05_10.jpg', 'ROI012_05_06.jpg']:
    img = np.array(Image.open(f'E:/aic/ihc-data/val/HLA-DR/{name}').convert('RGB'))
    print(f"{name}: shape={img.shape} mean={img.mean():.1f} "
          f"R={img[..., 0].mean():.1f} G={img[..., 1].mean():.1f} B={img[..., 2].mean():.1f} "
          f"min={img.min()} max={img.max()} std={img.std():.1f}")

print("\n=== val/HLA-DR (large ones from train_subset) ===")
for name in ['ROI000_00_00.jpg']:
    img = np.array(Image.open(f'E:/aic/ihc-data/train_subset/train/HLA-DR/{name}').convert('RGB'))
    print(f"{name}: shape={img.shape} mean={img.mean():.1f} "
          f"R={img[..., 0].mean():.1f} G={img[..., 1].mean():.1f} B={img[..., 2].mean():.1f} "
          f"min={img.min()} max={img.max()} std={img.std():.1f}")
