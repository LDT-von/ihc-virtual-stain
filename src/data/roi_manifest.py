"""Immutable ROI-grouped splits; the original dataset is never copied or moved."""
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MARKERS = ('HLA-DR', 'CD68', 'CD45RO', 'Vimentin')


def roi_id(name):
    match = re.fullmatch(r'(ROI\d+)_\d+_\d+\.jpg', name, re.IGNORECASE)
    if not match:
        raise ValueError(f'Unrecognized ROI filename: {name}')
    return match.group(1).upper()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_manifest(manifest):
    if manifest.get('format') != 1 or tuple(manifest.get('markers', ())) != MARKERS:
        raise ValueError('Unsupported manifest / marker order')
    if manifest.get('sha256') != digest({k: v for k, v in manifest.items() if k != 'sha256'}):
        raise ValueError('Manifest checksum mismatch')
    seen_names, seen_rois = set(), set()
    for split in ('train', 'val', 'holdout'):
        names = manifest['splits'][split]
        groups = {roi_id(n) for n in names}
        if not names or len(names) != len(set(names)):
            raise ValueError(f'Empty split or duplicate names: {split}')
        if seen_names.intersection(names) or seen_rois.intersection(groups):
            raise ValueError('ROI/sample leakage between splits')
        seen_names.update(names)
        seen_rois.update(groups)


def build_manifest(root, destination, seed=42, val_rois=3, holdout_rois=3):
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise FileExistsError(f'Will not overwrite split: {destination}')
    names = sorted(p.name for p in (root/'train'/'DAPI').glob('*.jpg'))
    if not names:
        raise ValueError('No training DAPI images')
    for marker in MARKERS:
        targets = {p.name for p in (root/'train'/marker).glob('*.jpg')}
        if set(names) != targets:
            raise ValueError(f'Incomplete or extra target pairs: {marker}')
    groups = sorted({roi_id(n) for n in names})
    if min(val_rois, holdout_rois) < 1 or val_rois+holdout_rois >= len(groups):
        raise ValueError('Insufficient ROI groups for train/val/holdout')
    random.Random(seed).shuffle(groups)
    val, holdout = set(groups[:val_rois]), set(groups[val_rois:val_rois+holdout_rois])
    manifest = {'format': 1, 'seed': seed, 'markers': list(MARKERS),
                'group_unit': 'ROI; patient independence unknown',
                'inventory_names_sha256': digest(names),
                'splits': {'train': [n for n in names if roi_id(n) not in val|holdout],
                           'val': [n for n in names if roi_id(n) in val],
                           'holdout': [n for n in names if roi_id(n) in holdout]}}
    manifest['sha256'] = digest(manifest)
    validate_manifest(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def read_gray(path):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert('RGB'))
    if not np.array_equal(rgb[..., 0], rgb[..., 1]) or not np.array_equal(rgb[..., 0], rgb[..., 2]):
        raise ValueError(f'Non-grayscale image: {path}; do not silently discard stain color')
    return np.ascontiguousarray(rgb[..., 0])


class PairedMarkers(Dataset):
    def __init__(self, root, names, augment=False, cache=True):
        self.root, self.names, self.augment = Path(root), list(names), augment
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('Empty or duplicate dataset')
        for name in self.names:
            roi_id(name)
        self.cache = None
        if cache:
            with ThreadPoolExecutor(6) as pool:
                self.cache = list(pool.map(self.read_pair, self.names))

    def read_pair(self, name):
        arrays = [read_gray(self.root/'train'/marker/name) for marker in ('DAPI',)+MARKERS]
        if any(a.shape != arrays[0].shape for a in arrays):
            raise ValueError(f'Unaligned shape: {name}')
        if arrays[0].shape != (256, 256):
            raise ValueError(f'Expected official 256x256 patch: {name}')
        return np.stack(arrays)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        arr = self.cache[index] if self.cache is not None else self.read_pair(self.names[index])
        # Isometric transforms preserve all marker intensities and pixel alignment.
        if self.augment:
            arr = np.rot90(arr, random.randrange(4), axes=(-2, -1))
            if random.random() < .5:
                arr = arr[..., ::-1]
        tensor = torch.from_numpy(np.array(arr, copy=True)).float().div_(255)
        return tensor[:1], tensor[1:], self.names[index]


def selected_names(names, limit, seed):
    if not limit or limit >= len(names):
        return list(names)
    if limit < 1:
        raise ValueError('Subset limit must be positive')
    return sorted(random.Random(seed).sample(list(names), limit))
