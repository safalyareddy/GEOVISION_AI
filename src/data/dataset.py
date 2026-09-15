"""
GeoVision AI Satellite Water Body Segmentation Platform
C7: Dataset Loading and Utility Layer

This module provides dataset loading abstractions and PyTorch Dataset / DataLoader
utilities for consuming the train, validation, and test dataset splits created in C6.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Callable, Sequence, Union

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:  # type: ignore
        pass
    DataLoader = Any  # type: ignore

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_dataset_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load dataset configuration from YAML config file, falling back to defaults if missing.
    """
    default_config: Dict[str, Any] = {
        "raw_dir": "dataset/raw",
        "prepared_dir": "dataset/prepared",
        "output_dir": "dataset",
        "image_dir_name": "images",
        "mask_dir_name": "masks",
        "supported_extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        "splits": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "seed": 42
        }
    }

    if config_path is None:
        config_path = Path("config/dataset.yaml")
        if not config_path.exists():
            config_path = Path("config/config.yaml")
    else:
        config_path = Path(config_path)

    if config_path.exists() and HAS_YAML:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict):
                dataset_cfg = data.get("dataset", {})
                if isinstance(dataset_cfg, dict):
                    for k in ["raw_dir", "prepared_dir", "output_dir", "image_dir_name", "mask_dir_name", "supported_extensions", "splits"]:
                        if k in dataset_cfg:
                            default_config[k] = dataset_cfg[k]
        except Exception as e:
            print(f"[WARNING] Failed to load dataset config from {config_path}: {e}")

    return default_config


class SatelliteWaterDataset(Dataset):
    """
    PyTorch Dataset implementation for Sentinel-2 satellite imagery and water body segmentation masks.
    Consumes the split dataset (train, val, test) structured by C6.
    """

    VALID_SPLITS: Tuple[str, ...] = ("train", "val", "test")

    def __init__(
        self,
        root_dir: Union[str, Path] = "dataset",
        split: str = "train",
        image_dir_name: str = "images",
        mask_dir_name: str = "masks",
        supported_extensions: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        to_tensor: bool = True
    ) -> None:
        """
        Initialize SatelliteWaterDataset.

        Args:
            root_dir: Base directory containing split subdirectories (train, val, test).
            split: Dataset split name ('train', 'val', or 'test').
            image_dir_name: Subfolder name for images (default: 'images').
            mask_dir_name: Subfolder name for masks (default: 'masks').
            supported_extensions: File extensions to include.
            transform: Optional transformation for images.
            target_transform: Optional transformation for masks.
            to_tensor: If True, convert loaded PIL images to PyTorch Tensors by default.
        """
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Split must be one of {self.VALID_SPLITS}.")

        self.root_dir = Path(root_dir)
        self.split = split
        self.image_dir_name = image_dir_name
        self.mask_dir_name = mask_dir_name
        self.transform = transform
        self.target_transform = target_transform
        self.to_tensor = to_tensor

        if supported_extensions is None:
            self.supported_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        else:
            self.supported_extensions = [ext.lower() for ext in supported_extensions]

        self.split_dir = self.root_dir / self.split
        self.images_dir = self.split_dir / self.image_dir_name
        self.masks_dir = self.split_dir / self.mask_dir_name

        self.pairs: List[Tuple[Path, Path]] = []
        self._discover_and_validate_pairs()

    def _discover_and_validate_pairs(self) -> None:
        """
        Discover image and mask files in the split directory and validate 1-to-1 pair matching.
        """
        if not self.split_dir.exists() or not self.split_dir.is_dir():
            raise FileNotFoundError(
                f"Split directory '{self.split_dir}' does not exist. "
                "Ensure C6 dataset splitting has been executed."
            )

        if not self.images_dir.exists() or not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory '{self.images_dir}' does not exist in split '{self.split}'."
            )

        if not self.masks_dir.exists() or not self.masks_dir.is_dir():
            raise FileNotFoundError(
                f"Mask directory '{self.masks_dir}' does not exist in split '{self.split}'."
            )

        ext_set = set(self.supported_extensions)

        img_map: Dict[str, Path] = {}
        for p in sorted(self.images_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in ext_set:
                img_map[p.stem] = p

        msk_map: Dict[str, Path] = {}
        for p in sorted(self.masks_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in ext_set:
                msk_map[p.stem] = p

        common_stems = sorted(list(set(img_map.keys()) & set(msk_map.keys())))
        unmatched_imgs = set(img_map.keys()) - set(msk_map.keys())
        unmatched_msks = set(msk_map.keys()) - set(img_map.keys())

        if unmatched_imgs or unmatched_msks:
            raise ValueError(
                f"Detected unmatched image/mask pairs in split '{self.split}'. "
                f"Unmatched images: {len(unmatched_imgs)}, Unmatched masks: {len(unmatched_msks)}."
            )

        if not common_stems:
            raise FileNotFoundError(
                f"No valid image-mask pairs found in split directory '{self.split_dir}'."
            )

        self.pairs = [(img_map[stem], msk_map[stem]) for stem in common_stems]

    def __len__(self) -> int:
        """Return the number of image-mask pairs in the split."""
        return len(self.pairs)

    def get_pair_paths(self, index: int) -> Tuple[Path, Path]:
        """
        Get the image and mask file paths for a given index.
        """
        if index < 0 or index >= len(self.pairs):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self.pairs)}.")
        return self.pairs[index]

    def load_image(self, img_path: Path) -> Any:
        """
        Load satellite image file into PIL Image (RGB format).
        """
        with Image.open(img_path) as img:
            return img.convert("RGB")

    def load_mask(self, msk_path: Path) -> Any:
        """
        Load water body segmentation mask file into PIL Image (L mode).
        """
        with Image.open(msk_path) as msk:
            return msk.convert("L")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get dataset sample at specified index.

        Returns:
            Dictionary containing:
                - 'image': Image tensor (or PIL Image if to_tensor=False)
                - 'mask': Mask tensor (or PIL Image if to_tensor=False)
                - 'stem': Filename stem string
                - 'image_path': Image file path string
                - 'mask_path': Mask file path string
        """
        img_path, msk_path = self.get_pair_paths(index)

        image = self.load_image(img_path)
        mask = self.load_mask(msk_path)

        if self.transform is not None:
            image = self.transform(image)
        elif self.to_tensor and HAS_TORCH:
            # Convert PIL RGB image (H, W, 3) to PyTorch FloatTensor (3, H, W) normalized to [0, 1]
            img_arr = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(img_arr).permute(2, 0, 1)

        if self.target_transform is not None:
            mask = self.target_transform(mask)
        elif self.to_tensor and HAS_TORCH:
            # Convert PIL L mask (H, W) to PyTorch FloatTensor (1, H, W) normalized to [0, 1]
            msk_arr = np.array(mask, dtype=np.float32)
            if msk_arr.max() > 1.0:
                msk_arr = msk_arr / 255.0
            mask = torch.from_numpy(msk_arr).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "stem": img_path.stem,
            "image_path": str(img_path),
            "mask_path": str(msk_path)
        }


def get_dataloader(
    split: str,
    root_dir: Union[str, Path] = "dataset",
    batch_size: int = 16,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    **kwargs: Any
) -> DataLoader:
    """
    Create and return a PyTorch DataLoader for the specified dataset split.

    Args:
        split: Dataset split ('train', 'val', or 'test').
        root_dir: Base directory containing split datasets.
        batch_size: Batch size for DataLoader.
        shuffle: Whether to shuffle samples. Defaults to True for 'train', False for others.
        num_workers: Number of worker subprocesses for data loading.
        **kwargs: Additional keyword arguments passed to DataLoader.

    Returns:
        PyTorch DataLoader instance.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to create a DataLoader instance.")

    if shuffle is None:
        shuffle = (split == "train")

    dataset = SatelliteWaterDataset(root_dir=root_dir, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
