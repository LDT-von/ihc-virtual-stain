"""扫描原始完整数据集的所有 marker，寻找"真实"IHC 图（非空白）"""
import numpy as np
from PIL import Image
from pathlib import Path

# 检查原始数据集所有 marker 的 train 目录
root = Path(r'E:\aic\初赛数据集（包含训练集和测试集输入）')
markers = ['HLA-DR', 'CD68', 'CD45RO', 'Vimentin']

for marker in markers:
    dapi_dir = root / 'train' / 'DAPI'
    ihc_dir = root / 'train' / marker
    
    if not dapi_dir.exists() or not ihc_dir.exists():
        print(f"{marker}: dirs not found")
        continue
    
    dapi_files = sorted(dapi_dir.glob('*.jpg'))
    ihc_files = sorted(ihc_dir.glob('*.jpg'))
    
    # 随机抽 20 张 IHC 统计
    means = []
    stds = []
    large_files = []
    for f in ihc_files[:100]:
        img = np.array(Image.open(f).convert('L'))
        means.append(img.mean())
        stds.append(img.std())
        if img.std() > 30:
            large_files.append((f.name, img.std(), img.mean()))
    
    print(f"\n{marker}: {len(ihc_files)} files")
    print(f"  DAPI mean={np.mean(means):.1f}±{np.std(means):.1f}")
    print(f"  IHC mean={np.mean(means):.1f}±{np.std(means):.1f}")
    print(f"  IHC std>30 (has structure): {len(large_files)} / 100")
    if large_files[:3]:
        print(f"  Top structure files: {large_files[:3]}")
    
    # 找最大的 5 张
    sizes = [(f.name, f.stat().st_size) for f in ihc_files]
    sizes.sort(key=lambda x: x[1], reverse=True)
    print(f"  Largest files: {sizes[:5]}")
    
    # 读最大文件看内容
    biggest_name = sizes[0][0]
    img = np.array(Image.open(ihc_dir / biggest_name).convert('RGB'))
    print(f"  Biggest ({biggest_name}): shape={img.shape}, mean={img.mean():.1f}, "
          f"R={img[...,0].mean():.1f} G={img[...,1].mean():.1f} B={img[...,2].mean():.1f}")
