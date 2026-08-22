"""
GeoVision AI Satellite Water Body Segmentation Platform
C6: Train, Validation, and Test Dataset Splitting Script

This script takes the prepared Sentinel-2 dataset created in C4 and partitions it
reproducibly into train, validation, and test splits according to configurable
ratios and a fixed random seed.
"""

import sys
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def load_dataset_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load dataset configuration from YAML configuration file.
    Falls back to sensible defaults if configuration is missing or incomplete.
    """
    default_config: Dict[str, Any] = {
        "raw_dir": "dataset/raw",
        "prepared_dir": "dataset/prepared",
        "output_dir": "dataset",
        "image_dir_name": "images",
        "mask_dir_name": "masks",
        "supported_extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"],
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "seed": 42
    }

    if config_path is None:
        config_path = Path("config/dataset.yaml")
        if not config_path.exists():
            config_path = Path("config/config.yaml")

    if config_path.exists():
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict):
                dataset_cfg = data.get("dataset", {})
                if isinstance(dataset_cfg, dict):
                    for key in ["raw_dir", "prepared_dir", "output_dir", "image_dir_name", "mask_dir_name", "supported_extensions"]:
                        if key in dataset_cfg:
                            default_config[key] = dataset_cfg[key]

                    splits_cfg = dataset_cfg.get("splits", {})
                    if isinstance(splits_cfg, dict):
                        for key in ["train_ratio", "val_ratio", "test_ratio", "seed"]:
                            if key in splits_cfg:
                                default_config[key] = splits_cfg[key]
        except ImportError:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("raw_dir:"):
                            default_config["raw_dir"] = line.split(":", 1)[1].strip().strip('"\'')
                        elif line.startswith("prepared_dir:"):
                            default_config["prepared_dir"] = line.split(":", 1)[1].strip().strip('"\'')
                        elif line.startswith("output_dir:"):
                            default_config["output_dir"] = line.split(":", 1)[1].strip().strip('"\'')
                        elif line.startswith("image_dir_name:"):
                            default_config["image_dir_name"] = line.split(":", 1)[1].strip().strip('"\'')
                        elif line.startswith("mask_dir_name:"):
                            default_config["mask_dir_name"] = line.split(":", 1)[1].strip().strip('"\'')
                        elif line.startswith("train_ratio:"):
                            default_config["train_ratio"] = float(line.split(":", 1)[1].strip())
                        elif line.startswith("val_ratio:"):
                            default_config["val_ratio"] = float(line.split(":", 1)[1].strip())
                        elif line.startswith("test_ratio:"):
                            default_config["test_ratio"] = float(line.split(":", 1)[1].strip())
                        elif line.startswith("seed:"):
                            default_config["seed"] = int(line.split(":", 1)[1].strip())
            except Exception as e:
                print(f"[WARNING] Failed to parse config file {config_path}: {e}. Using defaults.")
        except Exception as e:
            print(f"[WARNING] Failed to load config from {config_path}: {e}. Using defaults.")

    return default_config


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> bool:
    """
    Validate that train, validation, and test split ratios are valid non-negative numbers
    that sum to 1.0.
    """
    if train_ratio < 0.0 or val_ratio < 0.0 or test_ratio < 0.0:
        print(f"[ERROR] Split ratios must be non-negative. Got train={train_ratio}, val={val_ratio}, test={test_ratio}")
        return False

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-4:
        print(f"[ERROR] Split ratios must sum to 1.0. Sum of ({train_ratio} + {val_ratio} + {test_ratio}) is {ratio_sum:.4f}")
        return False

    return True


def discover_files(directory: Path, extensions: List[str]) -> Dict[str, Path]:
    """
    Discover files with supported extensions in the given directory.
    Returns a dictionary mapping filename stems to file Path objects in sorted deterministic order.
    """
    ext_set = {ext.lower() for ext in extensions}
    file_map: Dict[str, Path] = {}

    if not directory.exists() or not directory.is_dir():
        return file_map

    for filepath in sorted(directory.iterdir()):
        if filepath.is_file() and filepath.suffix.lower() in ext_set:
            file_map[filepath.stem] = filepath

    return file_map


def match_image_mask_pairs(
    image_files: Dict[str, Path],
    mask_files: Dict[str, Path]
) -> Tuple[List[Tuple[Path, Path]], List[Path], List[Path]]:
    """
    Match image files with corresponding mask files by filename stem.
    Returns matched pairs, unmatched images, and unmatched masks.
    """
    matched_pairs: List[Tuple[Path, Path]] = []
    unmatched_images: List[Path] = []
    unmatched_masks: List[Path] = []

    all_img_stems = set(image_files.keys())
    all_msk_stems = set(mask_files.keys())

    common_stems = sorted(list(all_img_stems & all_msk_stems))
    for stem in common_stems:
        matched_pairs.append((image_files[stem], mask_files[stem]))

    for stem in sorted(list(all_img_stems - all_msk_stems)):
        unmatched_images.append(image_files[stem])

    for stem in sorted(list(all_msk_stems - all_img_stems)):
        unmatched_masks.append(mask_files[stem])

    return matched_pairs, unmatched_images, unmatched_masks


def split_dataset_pairs(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Partition matched image-mask pairs reproducibly into train, val, and test splits
    using a fixed seed.
    """
    shuffled_pairs = list(pairs)

    # Use Random instance with fixed seed to ensure 100% reproducible shuffle
    rng = random.Random(seed)
    rng.shuffle(shuffled_pairs)

    total_count = len(shuffled_pairs)
    num_train = int(round(total_count * train_ratio))
    num_val = int(round(total_count * val_ratio))

    # Remaining samples assigned to test set to ensure total count is exact
    num_test = total_count - (num_train + num_val)

    train_pairs = shuffled_pairs[:num_train]
    val_pairs = shuffled_pairs[num_train:num_train + num_val]
    test_pairs = shuffled_pairs[num_train + num_val:]

    return {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs
    }


def split_dataset(
    config_path: Optional[Path] = None,
    seed_override: Optional[int] = None
) -> bool:
    """
    Execute dataset splitting on prepared Sentinel-2 data into train, val, and test splits.
    """
    print("=" * 60)
    print("GeoVision AI — Satellite Dataset Splitting (C6)")
    print("=" * 60)

    cfg = load_dataset_config(config_path)
    prepared_dir = Path(cfg["prepared_dir"])
    output_dir = Path(cfg["output_dir"])
    img_subfolder = cfg["image_dir_name"]
    msk_subfolder = cfg["mask_dir_name"]
    extensions = cfg["supported_extensions"]

    train_ratio = cfg["train_ratio"]
    val_ratio = cfg["val_ratio"]
    test_ratio = cfg["test_ratio"]
    seed = seed_override if seed_override is not None else cfg["seed"]

    print(f"[INFO] Configuration loaded:")
    print(f"       Prepared Dataset Source: {prepared_dir}")
    print(f"       Output Split Directory:  {output_dir}")
    print(f"       Split Ratios:            train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
    print(f"       Random Seed:             {seed}")

    # Step 1: Validate split ratios
    if not validate_split_ratios(train_ratio, val_ratio, test_ratio):
        return False

    # Step 2: Validate prepared dataset source directory existence
    if not prepared_dir.exists() or not prepared_dir.is_dir():
        print(f"[ERROR] Prepared dataset directory does not exist: '{prepared_dir}'")
        print("[ERROR] Prepared dataset is unavailable. Please run C4 preparation first.")
        return False

    prep_img_dir = prepared_dir / img_subfolder
    prep_msk_dir = prepared_dir / msk_subfolder

    if not prep_img_dir.exists() or not prep_img_dir.is_dir():
        print(f"[ERROR] Prepared image subfolder does not exist: '{prep_img_dir}'")
        return False

    if not prep_msk_dir.exists() or not prep_msk_dir.is_dir():
        print(f"[ERROR] Prepared mask subfolder does not exist: '{prep_msk_dir}'")
        return False

    # Step 3: Discover files & match pairs
    image_files = discover_files(prep_img_dir, extensions)
    mask_files = discover_files(prep_msk_dir, extensions)

    matched_pairs, unmatched_imgs, unmatched_msks = match_image_mask_pairs(image_files, mask_files)

    if unmatched_imgs:
        print(f"[WARNING] Detected {len(unmatched_imgs)} unmatched image file(s) in prepared dataset.")
    if unmatched_msks:
        print(f"[WARNING] Detected {len(unmatched_msks)} unmatched mask file(s) in prepared dataset.")

    if not matched_pairs:
        print(f"[ERROR] Cannot perform split: No valid image-mask pairs found in '{prepared_dir}'.")
        return False

    print(f"[INFO] Discovered {len(matched_pairs)} matched image-mask pair(s) in prepared dataset.")

    # Step 4: Perform reproducible dataset split
    split_dict = split_dataset_pairs(matched_pairs, train_ratio, val_ratio, test_ratio, seed)

    # Step 5: Copy split pairs to train, val, test subdirectories
    print(f"[INFO] Organizing split pairs into destination directories under '{output_dir}'...")
    total_copied = 0

    for split_name, pairs in split_dict.items():
        dst_img_dir = output_dir / split_name / img_subfolder
        dst_msk_dir = output_dir / split_name / msk_subfolder

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_msk_dir.mkdir(parents=True, exist_ok=True)

        for img_src, msk_src in pairs:
            shutil.copy2(img_src, dst_img_dir / img_src.name)
            shutil.copy2(msk_src, dst_msk_dir / msk_src.name)
            total_copied += 1

    # Step 6: Console summary report
    total_matched = len(matched_pairs)
    train_cnt = len(split_dict['train'])
    val_cnt = len(split_dict['val'])
    test_cnt = len(split_dict['test'])

    print("=" * 60)
    print("✂️ DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"  • Total Prepared Pairs Split: {total_matched}")
    print(f"  • Training Set (train):       {train_cnt} pairs ({(train_cnt / total_matched) * 100.0:.1f}%)")
    print(f"  • Validation Set (val):       {val_cnt} pairs ({(val_cnt / total_matched) * 100.0:.1f}%)")
    print(f"  • Test Set (test):            {test_cnt} pairs ({(test_cnt / total_matched) * 100.0:.1f}%)")
    print(f"  • Random Seed Used:           {seed}")
    print("=" * 60)
    print(f"[SUCCESS] Dataset successfully partitioned into train, val, and test splits!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = split_dataset()
    if not success:
        sys.exit(1)
