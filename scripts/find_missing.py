"""找出原始完整版 vs 精简版的差异文件"""
from pathlib import Path

full = Path(r'E:\aic\初赛数据集（包含训练集和测试集输入）\train\HLA-DR')
subset = Path(r'E:\aic\ihc-data\train\HLA-DR')

full_files = {f.name for f in full.glob('*.jpg')}
subset_files = {f.name for f in subset.glob('*.jpg')}

only_full = sorted(full_files - subset_files)
only_subset = sorted(subset_files - full_files)
print(f"Only in full (6296): {len(only_full)}")
print(f"Only in subset (5036): {len(only_subset)}")
print(f"\nFirst 10 only_full:")
for n in only_full[:10]:
    img_full = full / n
    img_subset = subset / n
    from PIL import Image
    import numpy as np
    if img_full.exists():
        arr = np.array(Image.open(img_full).convert('L'))
        print(f"  {n}: mean={arr.mean():.1f} std={arr.std():.1f} size={img_full.stat().st_size}")
    else:
        print(f"  {n}: NOT FOUND")

# 检查 only_full 里是否有结构
print("\nChecking only_full for real IHC content...")
from PIL import Image
import numpy as np

has_structure = 0
for name in only_full[:20]:
    f = full / name
    if f.exists():
        arr = np.array(Image.open(f).convert('L'))
        if arr.std() > 30:
            has_structure += 1
            print(f"  {name}: std={arr.std():.1f} HAS STRUCTURE!")
        else:
            print(f"  {name}: std={arr.std():.1f} (blank)")
print(f"\nStructure in only_full (first 20): {has_structure}/20")
