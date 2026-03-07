"""AffectNet dataset loaders for face emotion classification."""

import os

from torchvision import datasets, transforms

from .config import DATASET_DIR, INPUT_SIZE, MEAN, STD


def get_transforms(split: str) -> transforms.Compose:
    """Return image transforms for a given split."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


def rgb_loader(path: str):
    """Load image as RGB (handles grayscale / RGBA edge cases)."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def get_dataset(split: str, data_dir: str = DATASET_DIR) -> datasets.ImageFolder:
    """Return an ImageFolder dataset for the given split.

    Args:
        split    : "train", "val", or "test"
        data_dir : root directory containing train/, val/, test/ subdirs
    """
    return datasets.ImageFolder(
        root=os.path.join(data_dir, split),
        transform=get_transforms(split),
        loader=rgb_loader,
    )
