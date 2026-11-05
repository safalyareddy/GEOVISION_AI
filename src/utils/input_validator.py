"""
GeoVision AI — Input Validation & Geospatial Verification Engine

Enforces scientific safety protocols by distinguishing genuine Sentinel-2 satellite observations
from ordinary consumer photography, corrupted files, and unverified uploads.

Classification States:
- VERIFIED_PROJECT_SAMPLE: Matches curated Sentinel-2 test split tile with available ground truth.
- VERIFIED_GEOSPATIAL_INPUT: Contains GeoTIFF tags (CRS, ModelPixelScale, ModelTiepoint).
- USER_DECLARED_SATELLITE_UNVERIFIED: Plausible remote sensing image without verifiable geospatial metadata.
- UNSUPPORTED_PHOTOGRAPH: Ordinary consumer photo (flower, portrait, macro lens, smartphone EXIF).
- INVALID_IMAGE: Corrupted byte stream, unreadable payload, invalid dimensions.
- INFERENCE_FAILED: Runtime exception during inference execution.
"""

import io
import math
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("geovision.input_validator")


class InputClassification(str, Enum):
    VERIFIED_PROJECT_SAMPLE = "VERIFIED_PROJECT_SAMPLE"
    VERIFIED_GEOSPATIAL_INPUT = "VERIFIED_GEOSPATIAL_INPUT"
    USER_DECLARED_SATELLITE_UNVERIFIED = "USER_DECLARED_SATELLITE_UNVERIFIED"
    UNSUPPORTED_PHOTOGRAPH = "UNSUPPORTED_PHOTOGRAPH"
    INVALID_IMAGE = "INVALID_IMAGE"
    INFERENCE_FAILED = "INFERENCE_FAILED"


# Known project test sample hashes for fast, authoritative provenance verification
_PROJECT_SAMPLE_HASHES: Dict[str, str] = {}


def _init_project_sample_cache():
    """Cache hashes of curated dataset test split patches."""
    global _PROJECT_SAMPLE_HASHES
    if _PROJECT_SAMPLE_HASHES:
        return
    import hashlib
    test_dir = Path("dataset/test/images")
    if test_dir.exists():
        for p in test_dir.glob("*.png"):
            try:
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                _PROJECT_SAMPLE_HASHES[h] = p.name
            except Exception:
                pass


def classify_input_image(
    image: Image.Image,
    raw_bytes: Optional[bytes] = None,
    filename: Optional[str] = None
) -> Tuple[InputClassification, Dict[str, Any]]:
    """
    Classify input image into one of 6 safety categories and extract spatial/spectral metadata.

    Returns:
        (InputClassification, metadata_dict)
    """
    _init_project_sample_cache()

    meta: Dict[str, Any] = {
        "filename": filename or "unknown",
        "format": image.format,
        "mode": image.mode,
        "width": image.width,
        "height": image.height,
        "aspect_ratio": round(image.width / max(image.height, 1), 3),
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "has_geotiff_tags": False,
        "pixel_scale_m": None,
        "crs": None,
        "is_satellite_plausible": True,
        "rejection_reason": None,
    }

    # 1. Dimension and aspect ratio checks
    if image.width < 32 or image.height < 32:
        meta["rejection_reason"] = f"Image dimensions ({image.width}x{image.height}) are below minimum supported size (32x32)."
        return InputClassification.INVALID_IMAGE, meta

    if image.width > 8192 or image.height > 8192:
        meta["rejection_reason"] = f"Image dimensions ({image.width}x{image.height}) exceed maximum supported dimension (8192x8192)."
        return InputClassification.INVALID_IMAGE, meta

    aspect = meta["aspect_ratio"]
    if aspect < 0.2 or aspect > 5.0:
        meta["rejection_reason"] = f"Extreme aspect ratio ({aspect}:1) is incompatible with remote sensing tile processing."
        return InputClassification.INVALID_IMAGE, meta

    # 2. Check for exact Project Sample match (by bytes hash or known filename)
    if raw_bytes is not None:
        import hashlib
        h = hashlib.sha256(raw_bytes).hexdigest()
        if h in _PROJECT_SAMPLE_HASHES:
            meta["project_sample_name"] = _PROJECT_SAMPLE_HASHES[h]
            meta["is_project_sample"] = True
            return InputClassification.VERIFIED_PROJECT_SAMPLE, meta

    if filename and filename.startswith("patch_") and filename.endswith(".png"):
        known_path = Path("dataset/test/images") / filename
        if known_path.exists():
            meta["project_sample_name"] = filename
            meta["is_project_sample"] = True
            return InputClassification.VERIFIED_PROJECT_SAMPLE, meta

    # 3. Check EXIF tags for consumer photography signatures
    try:
        exif = image.getexif()
        if exif:
            meta["has_exif"] = True
            # EXIF Tag 271 is Make, 272 is Model
            make = exif.get(271) or exif.get(0x010f)
            model = exif.get(272) or exif.get(0x0110)
            if make:
                meta["camera_make"] = str(make).strip()
            if model:
                meta["camera_model"] = str(model).strip()

            # Consumer cameras: Apple, Canon, Nikon, Sony, Samsung, Google, etc.
            consumer_brands = ["apple", "canon", "nikon", "sony", "samsung", "google", "xiaomi", "huawei", "olympus", "fujifilm", "panasonic"]
            if make and any(b in str(make).lower() for b in consumer_brands):
                meta["rejection_reason"] = (
                    f"Consumer photograph detected (Camera: {meta['camera_make']} {meta.get('camera_model', '')}). "
                    "GeoVision AI is calibrated exclusively for Sentinel-2 optical satellite imagery."
                )
                return InputClassification.UNSUPPORTED_PHOTOGRAPH, meta
    except Exception:
        pass

    # 4. Check for GeoTIFF metadata tags (Tag 33550: ModelPixelScaleTag, 33922: ModelTiepointTag, 34735: GeoKeyDirectoryTag)
    try:
        if hasattr(image, "tag_v2"):
            tags = image.tag_v2
            # GeoKeyDirectoryTag or ModelPixelScaleTag
            if 33550 in tags or 34735 in tags:
                meta["has_geotiff_tags"] = True
                if 33550 in tags:
                    scale = tags[33550]
                    if len(scale) >= 2:
                        meta["pixel_scale_m"] = (float(scale[0]), float(scale[1]))
                return InputClassification.VERIFIED_GEOSPATIAL_INPUT, meta
    except Exception:
        pass

    # 5. Spectral & Photometric Heuristics for consumer photo rejection
    # Satellite imagery in natural color is normalized, smooth reflectance without extreme consumer vibrance/saturation
    try:
        rgb_img = image.convert("RGB")
        arr = np.array(rgb_img, dtype=np.float32) / 255.0  # (H, W, 3)

        # Compute saturation in HSV color space
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        saturation = np.where(cmax > 1e-5, delta / (cmax + 1e-5), 0.0)

        mean_sat = float(np.mean(saturation))
        p95_sat = float(np.percentile(saturation, 95))
        meta["mean_saturation"] = round(mean_sat, 3)
        meta["p95_saturation"] = round(p95_sat, 3)

        # Flower / Macro / Vivid Portrait heuristic:
        # Consumer photographs of flowers or indoor objects typically exhibit extreme saturation (>0.85) across large areas
        # and high channel disparity (e.g. vivid red flowers where R >> G and R >> B)
        vivid_saturated_fraction = float(np.mean(saturation > 0.80))
        meta["vivid_saturation_pct"] = round(vivid_saturated_fraction * 100.0, 2)

        if vivid_saturated_fraction > 0.35 and p95_sat > 0.90:
            meta["rejection_reason"] = (
                f"Unsupported photograph: Image exhibits extreme chromatic saturation ({meta['vivid_saturation_pct']}% of pixels > 80% saturation) "
                "characteristic of consumer floral, portrait, or indoor photography rather than multispectral satellite imagery."
            )
            return InputClassification.UNSUPPORTED_PHOTOGRAPH, meta

    except Exception as e:
        logger.warning(f"Spectral analysis error: {e}")

    # Default for generic, non-matching images: Allowed in unverified mode
    return InputClassification.USER_DECLARED_SATELLITE_UNVERIFIED, meta


def calculate_water_area_km2(
    water_pixels: int,
    classification: InputClassification,
    metadata: Dict[str, Any]
) -> Tuple[Optional[float], str]:
    """
    Calculate surface water area in square kilometers with rigorous scientific caveats.

    Returns:
        (area_km2, methodology_note)
    """
    if classification == InputClassification.UNSUPPORTED_PHOTOGRAPH:
        return None, "Calculations disabled: Input was classified as an unsupported consumer photograph."

    if classification == InputClassification.INVALID_IMAGE:
        return None, "Calculations disabled: Invalid or corrupted image."

    # 1. Verified Geospatial Mode (if GeoTIFF scale is present)
    if classification == InputClassification.VERIFIED_GEOSPATIAL_INPUT and metadata.get("pixel_scale_m"):
        scale_x, scale_y = metadata["pixel_scale_m"]
        pixel_area_m2 = abs(scale_x * scale_y)
        area_km2 = (water_pixels * pixel_area_m2) / 1e6
        note = (
            f"Verified Geospatial Calculation: Pixel scale {scale_x:.2f}m × {scale_y:.2f}m "
            f"derived directly from embedded GeoTIFF tags ({pixel_area_m2:.1f} m²/px)."
        )
        return round(area_km2, 4), note

    # 2. Curated Project Sample (Sentinel-2 10m nominal GSD)
    if classification == InputClassification.VERIFIED_PROJECT_SAMPLE:
        pixel_area_m2 = 100.0  # 10m Ground Sample Distance
        area_km2 = (water_pixels * pixel_area_m2) / 1e6
        note = (
            "Verified Project Sample: Calibrated Sentinel-2 Level-1C observation with nominal 10m GSD "
            "(100.0 m²/px nominal resolution assumption)."
        )
        return round(area_km2, 4), note

    # 3. User-Declared Satellite Unverified (Nominal Approximation Mode)
    pixel_area_m2 = 100.0
    area_km2 = (water_pixels * pixel_area_m2) / 1e6
    note = (
        "Nominal Approximation: Source imagery lacks verifiable geospatial coordinate reference or pixel scale tags. "
        "Calculated assuming a nominal Sentinel-2 10m GSD (100 m²/px). Not calibrated for surveying or legal boundaries."
    )
    return round(area_km2, 4), note
