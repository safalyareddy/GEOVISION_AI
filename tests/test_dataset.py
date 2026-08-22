"""
GeoVision AI Satellite Water Body Segmentation Platform
C7 Unit Tests: Dataset Loading and Utility Layer
"""

from pathlib import Path
from PIL import Image
import numpy as np

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

    class PytestRaisesContext:
        def __init__(self, expected_exception, match=None):
            self.expected_exception = expected_exception
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected exception {self.expected_exception.__name__} was not raised.")
            if not issubclass(exc_type, self.expected_exception):
                return False
            if self.match and self.match not in str(exc_val):
                raise AssertionError(f"Exception message '{exc_val}' did not match pattern '{self.match}'")
            return True

    class DummyPytest:
        @staticmethod
        def raises(expected_exception, match=None):
            return PytestRaisesContext(expected_exception, match=match)

        class mark:
            @staticmethod
            def skipif(condition, reason=None):
                def decorator(func):
                    return func
                return decorator

    pytest = DummyPytest()  # type: ignore

from src.data.dataset import (
    SatelliteWaterDataset,
    get_dataloader,
    load_dataset_config,
    HAS_TORCH
)


def create_synthetic_dataset(
    base_dir: Path,
    num_samples: int = 4,
    image_size: tuple = (32, 32),
    create_unmatched_img: bool = False,
    create_unmatched_msk: bool = False
) -> Path:
    """
    Helper fixture generator creating a synthetic dataset directory structure.
    """
    for split in ["train", "val", "test"]:
        img_dir = base_dir / split / "images"
        msk_dir = base_dir / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            stem = f"sample_{split}_{i:03d}"
            # Create synthetic RGB satellite image
            img_arr = np.random.randint(0, 256, size=(image_size[1], image_size[0], 3), dtype=np.uint8)
            img = Image.fromarray(img_arr, mode="RGB")
            img.save(img_dir / f"{stem}.png")

            # Create synthetic binary segmentation mask
            msk_arr = np.random.choice([0, 255], size=(image_size[1], image_size[0]), p=[0.7, 0.3]).astype(np.uint8)
            msk = Image.fromarray(msk_arr, mode="L")
            msk.save(msk_dir / f"{stem}.png")

        if create_unmatched_img:
            extra_img = Image.fromarray(np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8))
            extra_img.save(img_dir / f"extra_img_{split}.png")

        if create_unmatched_msk:
            extra_msk = Image.fromarray(np.zeros((image_size[1], image_size[0]), dtype=np.uint8))
            extra_msk.save(msk_dir / f"extra_msk_{split}.png")

    return base_dir


def test_load_dataset_config():
    """Verify loading configuration defaults and structure."""
    cfg = load_dataset_config()
    assert isinstance(cfg, dict)
    assert "raw_dir" in cfg
    assert "prepared_dir" in cfg
    assert "image_dir_name" in cfg
    assert "mask_dir_name" in cfg


def test_dataset_train_val_test_loading(tmp_path):
    """Verify train, val, and test splits can all be discovered and loaded."""
    dataset_dir = create_synthetic_dataset(tmp_path / "dataset", num_samples=3)

    for split in ["train", "val", "test"]:
        ds = SatelliteWaterDataset(root_dir=dataset_dir, split=split)
        assert len(ds) == 3
        assert ds.split == split


def test_dataset_item_access_and_shapes(tmp_path):
    """Verify dataset item access returns correct dictionary structure, stems, and shapes."""
    dataset_dir = create_synthetic_dataset(tmp_path / "dataset", num_samples=2, image_size=(16, 16))
    ds = SatelliteWaterDataset(root_dir=dataset_dir, split="train")

    sample = ds[0]
    assert isinstance(sample, dict)
    assert "image" in sample
    assert "mask" in sample
    assert "stem" in sample
    assert "image_path" in sample
    assert "mask_path" in sample

    assert sample["stem"] == "sample_train_000"

    if HAS_TORCH:
        import torch
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["mask"], torch.Tensor)
        assert sample["image"].shape == (3, 16, 16)
        assert sample["mask"].shape == (1, 16, 16)
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].dtype == torch.float32
        # Verify normalization
        assert sample["image"].max() <= 1.0
        assert sample["mask"].max() <= 1.0


def test_get_pair_paths(tmp_path):
    """Verify getting file paths for a given index."""
    dataset_dir = create_synthetic_dataset(tmp_path / "dataset", num_samples=2)
    ds = SatelliteWaterDataset(root_dir=dataset_dir, split="val")

    img_path, msk_path = ds.get_pair_paths(0)
    assert isinstance(img_path, Path)
    assert isinstance(msk_path, Path)
    assert img_path.name == "sample_val_000.png"
    assert msk_path.name == "sample_val_000.png"


def test_invalid_split_raises_value_error(tmp_path):
    """Verify invalid split name raises ValueError."""
    with pytest.raises(ValueError, match="Invalid split"):
        SatelliteWaterDataset(root_dir=tmp_path, split="invalid_split")


def test_missing_directory_raises_file_not_found_error(tmp_path):
    """Verify missing dataset directory raises FileNotFoundError."""
    non_existent_dir = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError):
        SatelliteWaterDataset(root_dir=non_existent_dir, split="train")


def test_unmatched_pairs_raises_value_error(tmp_path):
    """Verify unmatched image/mask files raise ValueError."""
    dataset_dir = create_synthetic_dataset(tmp_path / "dataset", num_samples=2, create_unmatched_img=True)
    with pytest.raises(ValueError, match="unmatched image/mask pairs"):
        SatelliteWaterDataset(root_dir=dataset_dir, split="train")


def test_out_of_range_index_raises_index_error(tmp_path):
    """Verify out of range index raises IndexError."""
    dataset_dir = create_synthetic_dataset(tmp_path / "dataset", num_samples=2)
    ds = SatelliteWaterDataset(root_dir=dataset_dir, split="train")

    with pytest.raises(IndexError):
        _ = ds[99]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_get_dataloader_creation(tmp_path):
    """Verify get_dataloader returns a valid PyTorch DataLoader."""
    dataset_dir = create_synthetic_dataset(tmp_path / "dataset", num_samples=4)
    loader = get_dataloader(split="train", root_dir=dataset_dir, batch_size=2)

    batch = next(iter(loader))
    assert "image" in batch
    assert "mask" in batch
    assert batch["image"].shape == (2, 3, 32, 32)
    assert batch["mask"].shape == (2, 1, 32, 32)
