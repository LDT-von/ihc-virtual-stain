"""DAPI -> IHC 病理图像配对数据集

数据目录约定：
    data/
      train/
        DAPI/             <- 配对的 DAPI patch
        IHC_<marker>/     <- 对应 IHC patch（同名同尺寸）
      test/
        DAPI/             <- 仅输入
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def build_transforms(patch_size: int, augment: bool) -> A.Compose:
    if augment:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.3),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
                ToTensorV2(),
            ],
            additional_targets={"ihc": "image"},
        )
    return A.Compose(
        [
            A.Resize(patch_size, patch_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={"ihc": "image"},
    )


class UnpairedDAPIDataset(Dataset):
    """
    Unpaired DAPI→IHC 数据集（DAPI 和 IHC 各自独立采样）

    目录结构（与 paired 版本相同）：
        train/
            DAPI/          <- DAPI patch
            IHC_<marker>/  <- IHC patch（独立采样，不需要配对）
        val/
            DAPI/
            IHC_<marker>/

    CUT 训练时使用：每次 sample 从 DAPI 和 IHC 各自独立采样，
    打破 pixel-level 的配对约束。
    """

    def __init__(self,
        root: str | Path,
        marker: str,
        split: str = "train",
        patch_size: int = 256,
        augment: bool = True,
        val_split: float = 0.05,
    ):
        self.root = Path(root)
        self.marker = marker
        self.split = split
        self.patch_size = patch_size
        self.val_split = val_split

        # val 目录可能不存在，自动从 train 划分
        train_dir = self.root / "train"
        val_dir = self.root / "val"

        self.dapi_dir = train_dir / "DAPI"
        ihc_dir = f"IHC_{marker}"
        # 尝试 train/IHC_{marker}，再尝试 train/{marker}
        ihc_train = train_dir / ihc_dir
        ihc_marker = train_dir / marker

        self.ihc_dir = ihc_train if ihc_train.exists() else ihc_marker

        self.dapi_files = list_image_files(self.dapi_dir)
        self.ihc_files = list_image_files(self.ihc_dir)

        if not self.dapi_files:
            print(f"[Dataset] 警告：train/DAPI/ 无图像")
        if not self.ihc_files:
            print(f"[Dataset] 警告：IHC 目录无图像：{self.ihc_dir}")

        print(f"[Unpaired Dataset] {marker}: "
              f"DAPI={len(self.dapi_files)} IHC={len(self.ihc_files)}")

        self.dapi_transform = build_transforms(patch_size, augment and split == "train")
        self.ihc_transform = build_transforms(patch_size, augment and split == "train")

        # val 模式下：自动从 train 划分 val_split%
        if split == "val":
            import random
            rng = random.Random(42)
            n_dapi = len(self.dapi_files)
            n_ihc = len(self.ihc_files)
            effective = min(n_dapi, n_ihc)
            val_n = max(1, int(effective * val_split))
            val_n = min(val_n, effective - 1)
            val_indices = sorted(rng.sample(range(effective), val_n))
            self._val_indices = set(val_indices)
            self.len_dapi = val_n
            self.len_ihc = val_n
        else:
            self.len_dapi = len(self.dapi_files)
            self.len_ihc = len(self.ihc_files)

    def __len__(self) -> int:
        # 返回较大者（每个 batch 独立采样）
        return max(self.len_dapi, self.len_ihc)

    def __getitem__(self, idx: int):
        import random

        if hasattr(self, '_val_indices'):
            # val 模式：按划分索引
            dapi_idx = self._val_indices[idx]
            ihc_idx = self._val_indices[idx]
        else:
            # train 模式：各自独立采样
            dapi_idx = idx % self.len_dapi
            ihc_idx = idx % self.len_ihc
            # 如果两者长度不同，换一个随机偏移打破配对
            if self.len_dapi != self.len_ihc:
                ihc_idx = random.randint(0, self.len_ihc - 1)

        dapi_path = self.dapi_files[dapi_idx]
        ihc_path = self.ihc_files[ihc_idx]

        dapi_img = load_image(dapi_path)
        ihc_img = load_image(ihc_path)

        dapi = self.dapi_transform(image=dapi_img)["image"]
        ihc = self.ihc_transform(image=ihc_img)["image"]

        return {
            "A": dapi,   # 源域 (DAPI)
            "B": ihc,    # 目标域 (IHC)
            "A_path": str(dapi_path),
            "B_path": str(ihc_path),
            "name": dapi_path.stem,
        }


class DAPItoIHCDataset(Dataset):
    """DAPI -> IHC 配对数据集

    Args:
        root: 数据根目录
        marker: IHC 标记名，如 "HLA-DR"
        split: "train" / "val" / "test"
        patch_size: 输出 patch 尺寸
        augment: 是否训练增强
    """

    def __init__(
        self,
        root: str | Path,
        marker: str,
        split: str = "train",
        patch_size: int = 256,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.marker = marker
        self.split = split

        split_dir = self.root / split
        dapi_dir = split_dir / "DAPI"
        if split != "test":
            ihc_dir = split_dir / f"IHC_{marker}"
            if not ihc_dir.exists():
                alt = split_dir / marker
                ihc_dir = alt if alt.exists() else ihc_dir
        else:
            ihc_dir = None

        self.dapi_files = list_image_files(dapi_dir)
        if self.dapi_files:
            print(f"[Dataset] {split}/{marker}: 找到 {len(self.dapi_files)} 张 DAPI")

        self.pairs: list[tuple[Path, Optional[Path]]] = []
        for dapi_path in self.dapi_files:
            if ihc_dir is None or split == "test":
                self.pairs.append((dapi_path, None))
            else:
                ihc_path = ihc_dir / dapi_path.name
                if ihc_path.exists():
                    self.pairs.append((dapi_path, ihc_path))
                else:
                    print(f"[Dataset] 警告：缺失配对 {ihc_path}，跳过")
        if ihc_dir is not None and self.pairs:
            print(f"[Dataset] {split}/{marker}: 有效配对 {len(self.pairs)}")

        self.transform = build_transforms(patch_size, augment and split == "train")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        dapi_path, ihc_path = self.pairs[idx]
        dapi_img = load_image(dapi_path)

        if ihc_path is not None:
            ihc_img = load_image(ihc_path)
            transformed = self.transform(image=dapi_img, ihc=ihc_img)
            dapi = transformed["image"]
            ihc = transformed["ihc"]
            return {
                "A": dapi,   # 源域 (DAPI) — 兼容 CUT
                "B": ihc,    # 目标域 (IHC) — 兼容 CUT
                "dapi": dapi,  # 向后兼容
                "ihc": ihc,    # 向后兼容
                "name": dapi_path.stem,
            }

        transformed = self.transform(image=dapi_img)
        return {
            "dapi": transformed["image"],
            "ihc": torch.zeros(3, *dapi_img.shape[:2]),
            "name": dapi_path.stem,
        }
