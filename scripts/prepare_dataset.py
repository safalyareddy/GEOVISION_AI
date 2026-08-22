"""
GeoVision AI Satellite Water Body Segmentation Platform
C4: Satellite Dataset Preparation Script

This script provides a clean, reproducible preparation step that takes the
configured raw Sentinel-2 dataset and organizes it into a consistent prepared
dataset structure for downstream stages.
"""

import sys
import shutil
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
        "image_dir_name": "images",
        "mask_dir_name": "masks",
        "supported_extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
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
                    for key in default_config:
                        if key in dataset_cfg:
                            default_config[key] = dataset_cfg[key]
        except ImportError:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("raw_dir:"):
                            default_config["raw_dir"] = line.split(":", 1)[1].strip().strip('"\'')
                        elif line.startswith("prepared_dir:"):
                            default_config["prepared_dir"] = line.split(":", 1)[1].strip().strip('"\'')
                        elif line.startswith("image_dir_name:"):
                            default_config["image_dir_name"] = line.split(":", 1)[1].strip().strip('"\'')
                        elif line.startswith("mask_dir_name:"):
                            default_config["mask_dir_name"] = line.split(":", 1)[1].strip().strip('"\'')
            except Exception as e:
                print(f"[WARNING] Failed to parse config file {config_path}: {e}. Using defaults.")
        except Exception as e:
            print(f"[WARNING] Failed to load config from {config_path}: {e}. Using defaults.")

    return default_config


def discover_files(directory: Path, extensions: List[str]) -> Dict[str, Path]:
    """
    Discover files with supported extensions in the given directory.
    Returns a dictionary mapping filename stems (without extension) to file Path objects.
    Alphabetically sorted to guarantee deterministic discovery order.
    """
    ext_set = {ext.lower() for ext in extensions}
    file_map: Dict[str, Path] = {}

    if not directory.exists() or not directory.is_dir():
        return file_map

    for filepath in sorted(directory.iterdir()):
        if filepath.is_file() and filepath.suffix.lower() in ext_set:
            stem = filepath.stem
            file_map[stem] = filepath

    return file_map


def match_image_mask_pairs(
    image_files: Dict[str, Path],
    mask_files: Dict[str, Path]
) -> Tuple[List[Tuple[Path, Path]], List[Path], List[Path]]:
    """
    Match satellite image files with corresponding segmentation mask files by filename stem.
    Returns:
        - matched_pairs: List of (image_path, mask_path) tuples
        - unmatched_images: List of image paths with no corresponding mask
        - unmatched_masks: List of mask paths with no corresponding image
    """
    matched_pairs: List[Tuple[Path, Path]] = []
    unmatched_images: List[Path] = []
    unmatched_masks: List[Path] = []

    all_image_stems = set(image_files.keys())
    all_mask_stems = set(mask_files.keys())

    common_stems = sorted(list(all_image_stems & all_mask_stems))
    for stem in common_stems:
        matched_pairs.append((image_files[stem], mask_files[stem]))

    for stem in sorted(list(all_image_stems - all_mask_stems)):
        unmatched_images.append(image_files[stem])

    for stem in sorted(list(all_mask_stems - all_image_stems)):
        unmatched_masks.append(mask_files[stem])

    return matched_pairs, unmatched_images, unmatched_masks


def prepare_dataset(config_path: Optional[Path] = None) -> bool:
    """
    Execute raw dataset validation, file discovery, image-mask matching,
    and safe copying into prepared dataset structure.
    """
    print("=" * 60)
    print("GeoVision AI — Satellite Dataset Preparation (C4)")
    print("=" * 60)

    cfg = load_dataset_config(config_path)
    raw_dir = Path(cfg["raw_dir"])
    prepared_dir = Path(cfg["prepared_dir"])
    img_subfolder = cfg["image_dir_name"]
    msk_subfolder = cfg["mask_dir_name"]
    extensions = cfg["supported_extensions"]

    print(f"[INFO] Configuration loaded:")
    print(f"       Raw Dataset Source:      {raw_dir}")
    print(f"       Prepared Destination:    {prepared_dir}")
    print(f"       Supported Extensions:    {', '.join(extensions)}")

    # Step 1: Validate raw dataset source existence
    if not raw_dir.exists():
        print(f"[ERROR] Raw dataset source directory does not exist: '{raw_dir}'")
        print("[ERROR] Source dataset is unavailable. Please place raw Sentinel-2 data into the raw directory.")
        return False

    # Step 2: Validate raw dataset source is a directory
    if not raw_dir.is_dir():
        print(f"[ERROR] Configured raw dataset source path is not a directory: '{raw_dir}'")
        return False

    # Step 3: Locate subdirectories for images and masks if present, else fallback to raw_dir
    raw_img_dir = raw_dir / img_subfolder
    raw_msk_dir = raw_dir / msk_subfolder

    if raw_img_dir.exists() and raw_img_dir.is_dir() and raw_msk_dir.exists() and raw_msk_dir.is_dir():
        image_search_dir = raw_img_dir
        mask_search_dir = raw_msk_dir
    else:
        image_search_dir = raw_dir
        mask_search_dir = raw_dir

    print(f"[INFO] Searching for raw satellite images in: {image_search_dir}")
    print(f"[INFO] Searching for raw segmentation masks in: {mask_search_dir}")

    # Step 4: Discover files
    image_files = discover_files(image_search_dir, extensions)
    mask_files = discover_files(mask_search_dir, extensions)

    print(f"[INFO] Discovered {len(image_files)} image candidate(s) and {len(mask_files)} mask candidate(s).")

    if not image_files:
        print(f"[ERROR] No satellite image files found with supported extensions in '{image_search_dir}'.")
        return False

    if not mask_files:
        print(f"[ERROR] No segmentation mask files found with supported extensions in '{mask_search_dir}'.")
        return False

    # Step 5: Match images and masks deterministically & detect unmatched files
    matched_pairs, unmatched_images, unmatched_masks = match_image_mask_pairs(image_files, mask_files)

    if unmatched_images:
        print(f"[WARNING] Detected {len(unmatched_images)} unmatched image file(s):")
        for img_path in unmatched_images[:5]:
            print(f"          - {img_path.name}")
        if len(unmatched_images) > 5:
            print(f"          ... and {len(unmatched_images) - 5} more.")

    if unmatched_masks:
        print(f"[WARNING] Detected {len(unmatched_masks)} unmatched mask file(s):")
        for msk_path in unmatched_masks[:5]:
            print(f"          - {msk_path.name}")
        if len(unmatched_masks) > 5:
            print(f"          ... and {len(unmatched_masks) - 5} more.")

    if not matched_pairs:
        print(f"[ERROR] Failed to match any valid image-mask pairs in raw dataset source '{raw_dir}'.")
        return False

    print(f"[INFO] Deterministically matched {len(matched_pairs)} image-mask pair(s).")

    # Step 6: Create prepared dataset directory structure
    prep_img_dir = prepared_dir / img_subfolder
    prep_msk_dir = prepared_dir / msk_subfolder

    print(f"[INFO] Creating prepared dataset directory structure at '{prepared_dir}'...")
    prep_img_dir.mkdir(parents=True, exist_ok=True)
    prep_msk_dir.mkdir(parents=True, exist_ok=True)

    # Step 7: Copy paired files safely into prepared dataset structure
    print(f"[INFO] Copying paired files into prepared dataset structure (preserving originals)...")
    copied_count = 0
    for img_src, msk_src in matched_pairs:
        img_dst = prep_img_dir / img_src.name
        msk_dst = prep_msk_dir / msk_src.name

        shutil.copy2(img_src, img_dst)
        shutil.copy2(msk_src, msk_dst)
        copied_count += 1

    print(f"[SUCCESS] Prepared dataset successfully created with {copied_count} pair(s) in '{prepared_dir}'.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = prepare_dataset()
    if not success:
        sys.exit(1)
