import sys
sys.path.insert(0, 'e:/aic/ihc-virtual-stain')
from src.data.dataset import DAPItoIHCDataset, list_image_files
from pathlib import Path

root = Path('e:/aic/ihc-virtual-stain/初赛数据集（包含训练集和测试集输入）/初赛数据集（包含训练集和测试集输入）/train_subset')
print(f'root={root}')
print(f'root/train exists: {(root / "train").exists()}')
print(f'root/DAPI exists: {(root / "DAPI").exists()}')
print(f'root/train/DAPI exists: {(root / "train" / "DAPI").exists()}')

# Check direct DAPI dir
dapi_files = list_image_files(root / "DAPI")
print(f'DAPI files in root/DAPI: {len(dapi_files)}')

# What does dataset expects
split_dir = root / "train"
print(f'split_dir: {split_dir}')
print(f'split_dir exists: {split_dir.exists()}')
