"""Image datasets and mini-batch loaders for pill presence classification."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

#===================================================================================

CLASS_TO_IDX = {"not_filled": 0, "filled": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)

#===================================================================================

def build_transforms(image_size: int = 224, training: bool = False) -> Callable:
    """
    Keep the entire image; apply mild color augmentation only for training.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    
    operations = [transforms.Resize((image_size, image_size))]
    if training:
        operations.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]
    )

    return transforms.Compose(operations)

#===================================================================================

class PillImageDataset(Dataset):
    """Read ``root/{not_filled,filled}/**/*`` using an explicit label mapping."""

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.transform = transform if transform is not None else build_transforms()
        self.class_to_idx = dict(CLASS_TO_IDX)
        self.class_counts: dict[str, int] = {}
        self.samples: list[tuple[Path, int]] = []

        for class_name, label in self.class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Required class directory is missing: {class_dir}")
            
            paths = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            if not paths:
                raise ValueError(f"No supported images found in class directory: {class_dir}")
            
            self.class_counts[class_name] = len(paths)
            self.samples.extend((path, label) for path in paths)

    #---------------------------------

    def __len__(self) -> int:
        return len(self.samples)
    
    #---------------------------------

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, label = self.samples[index]
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read image: {path}") from exc
        return self.transform(image), torch.tensor(label, dtype=torch.float32)

#===================================================================================

def create_dataloaders(
    dataset_root: Union[str, Path],
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/val batches without mixing the two directory splits.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    root = Path(dataset_root).expanduser()

    train_dataset = PillImageDataset(root / "train", build_transforms(image_size, True))
    val_dataset = PillImageDataset(root / "val", build_transforms(image_size, False))

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }

    # train set
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )

    # val set
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1),
        **common,
    )

    return train_loader, val_loader

#===================================================================================