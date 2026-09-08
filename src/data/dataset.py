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
                A.RandomCrop(patch_size, patch_size) if False else A.NoOp(),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
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
        ihc_dir = split_dir / f"IHC_{marker}" if split != "test" else None

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
                    # 容错：如果 IHC 缺失，跳过
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
                "dapi": dapi,
                "ihc": ihc,
                "name": dapi_path.stem,
            }

        # 测试模式：仅 DAPI
        transformed = self.transform(image=dapi_img)
        return {
            "dapi": transformed["image"],
            "ihc": torch.zeros(3, *dapi_img.shape[:2]),  # 占位
            "name": dapi_path.stem,
        }