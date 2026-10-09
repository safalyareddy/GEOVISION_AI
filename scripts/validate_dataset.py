"""
GeoVision AI Satellite Water Body Segmentation Platform
C5: Satellite Dataset Validation and Statistics Script

This script inspects the prepared Sentinel-2 dataset produced by C4,
validates image/mask pairing consistency and spatial shape alignment,
and calculates dataset-level statistics including water vs land pixel distribution.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from PIL import Image
import numpy as np


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


def discover_files(directory: Path, extensions: List[str], subfolder: Optional[str] = None) -> Dict[str, Path]:
    """
    Discover files with supported extensions in the given directory or subfolders.
    Returns a dictionary mapping filename stems (without extension) to file Path objects.
    Alphabetically sorted to guarantee deterministic discovery order.
    """
    ext_set = {ext.lower() for ext in extensions}
    file_map: Dict[str, Path] = {}

    if not directory.exists() or not directory.is_dir():
        return file_map

    if subfolder:
        search_paths = list(directory.rglob(f"**/{subfolder}/*"))
    else:
        search_paths = list(directory.iterdir())

    for filepath in sorted(search_paths):
        if filepath.is_file() and filepath.suffix.lower() in ext_set:
            stem = filepath.stem
            file_map[stem] = filepath

    return file_map


def inspect_image_mask_pair(
    img_path: Path,
    msk_path: Path
) -> Tuple[bool, Tuple[int, int], Tuple[int, int], int, int]:
    """
    Inspect a single image-mask pair:
    - Verifies image and mask dimensions match (height, width).
    - Computes water (foreground > 127) and background (land <= 127) pixel counts.
    Returns:
        - shape_match (bool)
        - img_size (height, width)
        - mask_size (height, width)
        - water_pixel_count (int)
        - land_pixel_count (int)
    """
    with Image.open(img_path) as img:
        img_w, img_h = img.size
        img_size = (img_h, img_w)

    with Image.open(msk_path) as msk:
        msk_w, msk_h = msk.size
        msk_size = (msk_h, msk_w)
        msk_arr = np.array(msk)

    shape_match = (img_size == msk_size)

    # Class / Pixel distribution (pixels > 127 treated as water, <= 127 as background/land)
    water_pixels = int(np.sum(msk_arr > 127))
    land_pixels = int(msk_arr.size - water_pixels)

    return shape_match, img_size, msk_size, water_pixels, land_pixels


def validate_and_analyze_dataset(
    config_path: Optional[Path] = None,
    report_output_path: Optional[Path] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute dataset validation checks and statistics calculations on the prepared dataset.
    Returns:
        - success (bool)
        - stats (Dict[str, Any]) containing calculated metrics
    """
    print("=" * 60)
    print("GeoVision AI — Satellite Dataset Validation & Statistics (C5)")
    print("=" * 60)

    cfg = load_dataset_config(config_path)
    prepared_dir = Path(cfg["prepared_dir"])
    img_subfolder = cfg["image_dir_name"]
    msk_subfolder = cfg["mask_dir_name"]
    extensions = cfg["supported_extensions"]

    print(f"[INFO] Configuration loaded:")
    print(f"       Prepared Directory:      {prepared_dir}")
    print(f"       Supported Extensions:    {', '.join(extensions)}")

    # Step 1: Validate prepared dataset or split directories existence
    if not prepared_dir.exists():
        # Fall back to dataset root if train/val/test directories exist
        dataset_root = Path(cfg.get("dataset_root", "dataset"))
        if (dataset_root / "train").exists() and (dataset_root / "test").exists():
            print(f"[INFO] Prepared dir '{prepared_dir}' not found. Falling back to split dataset root '{dataset_root}'.")
            prepared_dir = dataset_root
        else:
            print(f"[ERROR] Prepared dataset directory does not exist: '{prepared_dir}'")
            print("[ERROR] Prepared dataset is unavailable. Please run C4 preparation first.")
            return False, {}

    if not prepared_dir.is_dir():
        print(f"[ERROR] Configured prepared dataset path is not a directory: '{prepared_dir}'")
        return False, {}

    # Check for split subdirectories vs flat image/mask subdirectories
    splits = ["train", "val", "test"]
    split_mode = all((prepared_dir / s / img_subfolder).exists() for s in splits)

    if split_mode:
        print(f"[INFO] Scanning split dataset in: {prepared_dir} ({', '.join(splits)})")
        prep_img_dir = prepared_dir
        prep_msk_dir = prepared_dir
    else:
        prep_img_dir = prepared_dir / img_subfolder
        prep_msk_dir = prepared_dir / msk_subfolder

        if not prep_img_dir.exists() or not prep_img_dir.is_dir():
            print(f"[ERROR] Prepared image subfolder does not exist: '{prep_img_dir}'")
            return False, {}

        if not prep_msk_dir.exists() or not prep_msk_dir.is_dir():
            print(f"[ERROR] Prepared mask subfolder does not exist: '{prep_msk_dir}'")
            return False, {}

        print(f"[INFO] Scanning prepared images in: {prep_img_dir}")
        print(f"[INFO] Scanning prepared masks in:  {prep_msk_dir}")

    # Step 2: Discover image and mask files
    if split_mode:
        image_files = discover_files(prepared_dir, extensions, subfolder=img_subfolder)
        mask_files = discover_files(prepared_dir, extensions, subfolder=msk_subfolder)
    else:
        image_files = discover_files(prep_img_dir, extensions)
        mask_files = discover_files(prep_msk_dir, extensions)

    print(f"[INFO] Found {len(image_files)} image file(s) and {len(mask_files)} mask file(s).")

    all_image_stems = set(image_files.keys())
    all_mask_stems = set(mask_files.keys())

    common_stems = sorted(list(all_image_stems & all_mask_stems))
    unmatched_images = sorted([image_files[s] for s in (all_image_stems - all_mask_stems)])
    unmatched_masks = sorted([mask_files[s] for s in (all_mask_stems - all_image_stems)])

    # Step 3: Report unmatched files
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

    if not common_stems:
        print(f"[ERROR] Validation failed: No matching image-mask pairs found in '{prepared_dir}'.")
        return False, {}

    print(f"[INFO] Found {len(common_stems)} matched image-mask pair(s). Analyzing spatial & pixel metrics...")

    # Step 4: Spatial shape validation & pixel statistics calculation
    shape_mismatches: List[Tuple[Path, Path, Tuple[int, int], Tuple[int, int]]] = []
    observed_img_shapes: Set[Tuple[int, int]] = set()
    observed_msk_shapes: Set[Tuple[int, int]] = set()
    total_water_pixels = 0
    total_land_pixels = 0

    for stem in common_stems:
        img_path = image_files[stem]
        msk_path = mask_files[stem]

        match_ok, img_shape, msk_shape, water_px, land_px = inspect_image_mask_pair(img_path, msk_path)
        observed_img_shapes.add(img_shape)
        observed_msk_shapes.add(msk_shape)
        total_water_pixels += water_px
        total_land_pixels += land_px

        if not match_ok:
            shape_mismatches.append((img_path, msk_path, img_shape, msk_shape))

    total_pixels = total_water_pixels + total_land_pixels
    water_pct = (total_water_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
    land_pct = (total_land_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

    # Step 5: Report shape mismatches if any
    if shape_mismatches:
        print(f"[ERROR] Validation failed: {len(shape_mismatches)} pair(s) have incompatible spatial dimensions:")
        for img_p, msk_p, i_s, m_s in shape_mismatches[:5]:
            print(f"        - {img_p.name} (img: {i_s}) vs {msk_p.name} (mask: {m_s})")
        return False, {}

    stats: Dict[str, Any] = {
        "total_images": len(image_files),
        "total_masks": len(mask_files),
        "paired_samples": len(common_stems),
        "unmatched_images": len(unmatched_images),
        "unmatched_masks": len(unmatched_masks),
        "observed_image_shapes": [f"{h}x{w}" for h, w in sorted(list(observed_img_shapes))],
        "observed_mask_shapes": [f"{h}x{w}" for h, w in sorted(list(observed_msk_shapes))],
        "total_pixels": total_pixels,
        "water_pixels": total_water_pixels,
        "land_pixels": total_land_pixels,
        "water_percentage": round(water_pct, 2),
        "land_percentage": round(land_pct, 2),
    }

    # Step 6: Console output reporting
    print("=" * 60)
    print("📊 DATASET VALIDATION & STATISTICS SUMMARY")
    print("=" * 60)
    print(f"  • Total Images Discovered:   {stats['total_images']}")
    print(f"  • Total Masks Discovered:    {stats['total_masks']}")
    print(f"  • Valid Paired Samples:      {stats['paired_samples']}")
    print(f"  • Unmatched Images:          {stats['unmatched_images']}")
    print(f"  • Unmatched Masks:           {stats['unmatched_masks']}")
    print(f"  • Image Dimensions (HxW):    {', '.join(stats['observed_image_shapes'])}")
    print(f"  • Mask Dimensions (HxW):     {', '.join(stats['observed_mask_shapes'])}")
    print("-" * 60)
    print(f"  • Class Distribution (Pixel-level):")
    print(f"    - Land / Background (Class 0): {stats['land_pixels']:,} px ({stats['land_percentage']}%)")
    print(f"    - Surface Water (Class 1):     {stats['water_pixels']:,} px ({stats['water_percentage']}%)")
    print("=" * 60)
    print("[SUCCESS] Dataset validation passed! All paired samples are consistent.")
    print("=" * 60)

    # Optional report writing if output path provided (not committed)
    if report_output_path is not None:
        try:
            report_str = f"# GeoVision AI — Dataset Validation & Statistics Report\n\n"
            report_str += f"- **Paired Samples**: `{stats['paired_samples']}`\n"
            report_str += f"- **Image Dimensions**: `{', '.join(stats['observed_image_shapes'])}`\n"
            report_str += f"- **Land / Background**: `{stats['land_percentage']}%` (`{stats['land_pixels']}` px)\n"
            report_str += f"- **Surface Water**: `{stats['water_percentage']}%` (`{stats['water_pixels']}` px)\n"
            with open(report_output_path, "w", encoding="utf-8") as f:
                f.write(report_str)
            print(f"[INFO] Report written to: {report_output_path}")
        except Exception as e:
            print(f"[WARNING] Failed to write report file: {e}")

    return True, stats


if __name__ == "__main__":
    success, _ = validate_and_analyze_dataset()
    if not success:
        sys.exit(1)
