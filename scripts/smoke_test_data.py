"""数据加载 smoke test"""
import sys
sys.path.insert(0, ".")

from src.data.dataset import DAPItoIHCDataset

DATA_ROOT = r"E:\aic\初赛数据集（包含训练集和测试集输入）"

print("=" * 60)
print("数据加载 smoke test")
print("=" * 60)

for marker in ["HLA-DR", "CD45RO", "Vimentin", "CD68"]:
    ds = DAPItoIHCDataset(
        root=DATA_ROOT, marker=marker, split="train", patch_size=256, augment=False
    )
    if len(ds) == 0:
        print(f"  {marker:10s}: 0 pairs (FAIL)")
        continue
    s = ds[0]
    print(
        f"  {marker:10s}: {len(ds)} pairs, "
        f"dapi={tuple(s['dapi'].shape)}, ihc={tuple(s['ihc'].shape)}, "
        f"name={s['name']}"
    )

print()
test_ds = DAPItoIHCDataset(
    root=DATA_ROOT, marker="HLA-DR", split="test", patch_size=256, augment=False
)
print(f"  test       : {len(test_ds)} samples, dapi={tuple(test_ds[0]['dapi'].shape)}")
print("=" * 60)