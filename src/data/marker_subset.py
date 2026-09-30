"""Select a fixed subset of marker targets without changing the paired reader."""

from torch.utils.data import Dataset

from .roi_manifest import MARKERS


class MarkerSubsetDataset(Dataset):
    """Preserve sample order and augmentation while selecting target channels."""

    def __init__(self, dataset, markers):
        names = tuple(markers)
        if (not names or len(set(names)) != len(names)
                or names != tuple(marker for marker in MARKERS if marker in names)):
            raise ValueError("Markers must be a nonempty canonical-order subset")
        self.dataset = dataset
        self.markers = names
        self.indices = tuple(MARKERS.index(marker) for marker in names)

    @property
    def aug_strength(self):
        return self.dataset.aug_strength

    @aug_strength.setter
    def aug_strength(self, value):
        self.dataset.aug_strength = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, targets, name = self.dataset[index]
        return image, targets[list(self.indices)], name
