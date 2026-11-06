"""
GeoVision AI — Input Validation & Geospatial Verification Engine

Enforces scientific safety protocols by distinguishing genuine Sentinel-2 satellite observations
from ordinary consumer photography, corrupted files, and unverified uploads.

Classification States:
- VERIFIED_PROJECT_SAMPLE: Matches cryptographic SHA-256 hash of curated Sentinel-2 test split patch.
- VERIFIED_GEOSPATIAL_INPUT: Contains valid GeoTIFF tags (CRS, ModelPixelScale/Transformation) with verified metric scaling.
- USER_DECLARED_SATELLITE_UNVERIFIED: Plausible remote sensing imagery without verifiable geospatial metadata tags.
- UNSUPPORTED_INPUT: Consumer photography (camera hardware EXIF, floral/indoor macros), solid blanks, or unsupported objects.
- INVALID_IMAGE: Corrupted byte stream, zero-byte file, oversized (>10MB), or invalid dimensions/aspect ratio.
- INFERENCE_FAILED: Runtime exception during inference execution.
"""

import io
import math
import hashlib
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
from PIL import Image

logger = logging.getLogger("geovision.input_validator")

# Enforce security constants
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # Strict 10 MB upload ceiling
Image.MAX_IMAGE_PIXELS = 100_000_000  # Guard against decompression bomb DOS attacks

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class InputClassification(str, Enum):
    VERIFIED_PROJECT_SAMPLE = "VERIFIED_PROJECT_SAMPLE"
    VERIFIED_GEOSPATIAL_INPUT = "VERIFIED_GEOSPATIAL_INPUT"
    USER_DECLARED_SATELLITE_UNVERIFIED = "USER_DECLARED_SATELLITE_UNVERIFIED"
    UNSUPPORTED_INPUT = "UNSUPPORTED_INPUT"
    UNSUPPORTED_PHOTOGRAPH = "UNSUPPORTED_INPUT"  # Backward-compatible alias
    INVALID_IMAGE = "INVALID_IMAGE"
    INFERENCE_FAILED = "INFERENCE_FAILED"


# Known project test sample hashes for fast, authoritative provenance verification
_PROJECT_SAMPLE_HASHES: Dict[str, str] = {
    "ae6b621438ba1574c511ca51fabef1bb3f7347e90e63bf3543e60d046a8c4656": "patch_0047.png",
}


def _init_project_sample_cache():
    """Cache SHA-256 hashes of curated dataset test split patches."""
    global _PROJECT_SAMPLE_HASHES
    test_dirs = [Path("dataset/test/images"), Path("assets/demo")]
    for tdir in test_dirs:
        if tdir.exists():
            for p in tdir.glob("*.png"):
                try:
                    h = hashlib.sha256(p.read_bytes()).hexdigest()
                    _PROJECT_SAMPLE_HASHES[h] = p.name
                except Exception:
                    pass


def validate_raw_payload(
    raw_bytes: Optional[bytes] = None,
    filename: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate raw payload size, file extension, and basic magic headers before decoding.
    """
    if raw_bytes is not None:
        if len(raw_bytes) == 0:
            return False, "Zero-byte payload: Uploaded file is completely empty."
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            size_mb = len(raw_bytes) / (1024 * 1024)
            return False, f"Oversized upload: File size ({size_mb:.2f} MB) exceeds maximum allowed 10 MB limit."

    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file extension '{ext}'. Only PNG, JPEG, and TIFF are supported."

    # Validate image magic bytes if raw_bytes available
    if raw_bytes is not None and len(raw_bytes) >= 4:
        is_png = raw_bytes.startswith(b"\x89PNG")
        is_jpg = raw_bytes.startswith(b"\xff\xd8\xff")
        is_tif = raw_bytes.startswith(b"II*\x00") or raw_bytes.startswith(b"MM\x00*")
        if not (is_png or is_jpg or is_tif):
            return False, "Invalid MIME / content mismatch: Byte stream does not match valid PNG, JPEG, or TIFF headers."

    return True, None


def classify_input_image(
    image: Optional[Image.Image] = None,
    raw_bytes: Optional[bytes] = None,
    filename: Optional[str] = None
) -> Tuple[InputClassification, Dict[str, Any]]:
    """
    Classify input image into one of the authoritative safety categories and extract
    spatial/spectral metadata without relying on unvalidated assumptions.

    Returns:
        (InputClassification, metadata_dict)
    """
    _init_project_sample_cache()

    meta: Dict[str, Any] = {
        "filename": filename or "unknown",
        "format": getattr(image, "format", None) if image else None,
        "mode": getattr(image, "mode", None) if image else None,
        "width": getattr(image, "width", 0) if image else 0,
        "height": getattr(image, "height", 0) if image else 0,
        "aspect_ratio": 1.0,
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "has_geotiff_tags": False,
        "has_crs": False,
        "crs_info": None,
        "pixel_scale_m": None,
        "pixel_area_m2": None,
        "coordinates_in_degrees": False,
        "is_satellite_plausible": True,
        "rejection_reason": None,
        "unverified_disclaimer": None,
    }

    # 1. Server-side Payload & Size Validation
    is_valid_payload, payload_err = validate_raw_payload(raw_bytes, filename)
    if not is_valid_payload:
        meta["rejection_reason"] = payload_err
        return InputClassification.INVALID_IMAGE, meta

    # 2. Decode Image if not already provided
    if image is None and raw_bytes is not None:
        try:
            image = Image.open(io.BytesIO(raw_bytes))
        except Exception as e:
            meta["rejection_reason"] = f"Corrupted or unreadable image file: {type(e).__name__} ({e})"
            return InputClassification.INVALID_IMAGE, meta

    if image is None:
        meta["rejection_reason"] = "No image payload provided."
        return InputClassification.INVALID_IMAGE, meta

    meta["format"] = image.format
    meta["mode"] = image.mode
    meta["width"] = image.width
    meta["height"] = image.height
    meta["aspect_ratio"] = round(image.width / max(image.height, 1), 3)

    # 3. Dimension and Aspect Ratio Constraints
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

    # 4. Check for Blank White, Blank Black, or Solid Uniform Images & Extreme Floral Saturation
    try:
        rgb_thumb = image.convert("RGB")
        # Sample or check downscaled array for fast variance check
        sample_arr = np.array(rgb_thumb.resize((min(image.width, 128), min(image.height, 128))), dtype=np.float32)
        spatial_std = float(np.mean(np.std(sample_arr, axis=(0, 1))))
        mean_intensity = float(np.mean(sample_arr))
        # Specifically reject Blank Black (mean < 2.0) or Blank White (mean > 253.0)
        if spatial_std < 0.5 and (mean_intensity < 2.0 or mean_intensity > 253.0):
            blank_type = "black" if mean_intensity < 128.0 else "white"
            meta["rejection_reason"] = (
                f"Blank or uniform image detected (blank {blank_type} image, spatial standard deviation: {spatial_std:.2f}). "
                "Satellite segmentation requires valid multispectral remote sensing content."
            )
            return InputClassification.UNSUPPORTED_INPUT, meta

        # Compute saturation in HSV space for consumer floral / portrait rejection
        arr_norm = sample_arr / 255.0
        r, g, b = arr_norm[..., 0], arr_norm[..., 1], arr_norm[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        saturation = np.where(cmax > 1e-5, delta / (cmax + 1e-5), 0.0)

        mean_sat = float(np.mean(saturation))
        p95_sat = float(np.percentile(saturation, 95))
        vivid_saturated_fraction = float(np.mean(saturation > 0.80))
        meta["mean_saturation"] = round(mean_sat, 3)
        meta["p95_saturation"] = round(p95_sat, 3)
        meta["vivid_saturation_pct"] = round(vivid_saturated_fraction * 100.0, 2)

        consumer_keywords = ["flower", "portrait", "selfie", "food", "animal", "indoor", "room", "macro", "iphone", "camera"]
        has_consumer_name = any(k in (filename or "").lower() for k in consumer_keywords)

        if vivid_saturated_fraction > 0.35 and p95_sat > 0.90:
            if has_consumer_name:
                meta["rejection_reason"] = (
                    f"Unsupported photograph: Image exhibits extreme chromatic saturation ({meta['vivid_saturation_pct']}% of pixels > 80% saturation) "
                    "characteristic of consumer floral, portrait, or indoor photography rather than multispectral satellite imagery."
                )
                return InputClassification.UNSUPPORTED_INPUT, meta
            else:
                meta["saturation_advisory"] = (
                    f"Advisory: High chromatic saturation detected ({meta['vivid_saturation_pct']}% of pixels > 80% saturation). "
                    "Please verify whether this is an Earth observation false-color composite or an unverified upload."
                )
    except Exception as e:
        logger.warning(f"Spatial/spectral analysis error: {e}")

    # 5. Check for exact Project Sample match (Authoritative Cryptographic Hash ONLY)
    image_hash = None
    if raw_bytes is not None:
        image_hash = hashlib.sha256(raw_bytes).hexdigest()
    else:
        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_hash = hashlib.sha256(buf.getvalue()).hexdigest()
        except Exception:
            pass

    if image_hash and image_hash in _PROJECT_SAMPLE_HASHES:
        meta["project_sample_name"] = _PROJECT_SAMPLE_HASHES[image_hash]
        meta["is_project_sample"] = True
        return InputClassification.VERIFIED_PROJECT_SAMPLE, meta

    # 6. Check for GeoTIFF metadata tags (CRS, ModelPixelScale, ModelTiepoint, ModelTransformation)
    try:
        if hasattr(image, "tag_v2"):
            tags = image.tag_v2
            has_pixel_scale = 33550 in tags
            has_tiepoint = 33922 in tags
            has_transform = 34264 in tags
            has_geokey = 34735 in tags

            if has_pixel_scale or has_transform or has_geokey:
                meta["has_geotiff_tags"] = True

                # Check CRS validity
                if not has_geokey:
                    meta["crs_info"] = "Missing GeoKeyDirectoryTag (CRS missing)"
                    meta["has_crs"] = False
                else:
                    geokeys = tags[34735]
                    # GeoKey directory header must have at least 4 items
                    if len(geokeys) >= 4:
                        meta["has_crs"] = True
                        meta["crs_info"] = "Embedded GeoTIFF GeoKey Directory"
                    else:
                        meta["has_crs"] = False
                        meta["crs_info"] = "Malformed GeoKey directory"

                # Check Pixel Scale
                scale_valid = False
                if has_pixel_scale:
                    scale = tags[33550]
                    if len(scale) >= 2 and scale[0] > 0 and scale[1] > 0 and not math.isnan(scale[0]):
                        sx, sy = float(scale[0]), float(scale[1])
                        # Detect geographic degrees (e.g. WGS84 lat/long 0.0001 degrees)
                        if sx < 0.05 and sy < 0.05:
                            meta["coordinates_in_degrees"] = True
                            # Compute latitude from tiepoint if present
                            lat = 0.0
                            if has_tiepoint and len(tags[33922]) >= 5:
                                lat = float(tags[33922][4])
                            cos_lat = max(math.cos(math.radians(lat)), 0.01)
                            # 1 degree latitude ~ 111,320m; 1 degree longitude ~ 111,320m * cos(lat)
                            sx_m = sx * 111320.0 * cos_lat
                            sy_m = sy * 111320.0
                            meta["pixel_scale_m"] = (round(sx_m, 2), round(sy_m, 2))
                            meta["pixel_area_m2"] = sx_m * sy_m
                        else:
                            meta["pixel_scale_m"] = (sx, sy)
                            meta["pixel_area_m2"] = sx * sy
                        scale_valid = True

                # Handle affine transformation matrix (handles rotated pixels)
                if not scale_valid and has_transform:
                    transform = tags[34264]
                    if len(transform) >= 16:
                        # 4x4 affine matrix: m00=transform[0], m01=transform[1], m10=transform[4], m11=transform[5]
                        a, b = float(transform[0]), float(transform[1])
                        c, d = float(transform[4]), float(transform[5])
                        det = abs(a * d - b * c)
                        if det > 0 and not math.isnan(det):
                            meta["pixel_area_m2"] = det
                            meta["pixel_scale_m"] = (round(math.sqrt(det), 2), round(math.sqrt(det), 2))
                            scale_valid = True

                # Verified Geospatial Input requires BOTH valid CRS and valid positive pixel scale
                if meta["has_crs"] and scale_valid and meta["pixel_area_m2"] is not None:
                    return InputClassification.VERIFIED_GEOSPATIAL_INPUT, meta
                else:
                    logger.info("GeoTIFF detected but lacks complete CRS or valid pixel scale. Falling back to unverified mode.")
    except Exception as e_geo:
        logger.warning(f"GeoTIFF parsing diagnostic: {e_geo}")

    # 7. Check EXIF for consumer camera hardware signatures
    try:
        exif = image.getexif()
        if exif:
            make = exif.get(271) or exif.get(0x010F)
            model = exif.get(272) or exif.get(0x0110)
            if make:
                meta["camera_make"] = str(make).strip()
                meta["has_exif"] = True
            if model:
                meta["camera_model"] = str(model).strip()
                meta["has_exif"] = True

            consumer_brands = [
                "apple", "canon", "nikon", "sony", "samsung", "google",
                "xiaomi", "huawei", "olympus", "fujifilm", "panasonic", "leica", "motorola"
            ]
            if make and any(b in str(make).lower() for b in consumer_brands):
                meta["rejection_reason"] = (
                    f"Consumer photograph detected (Camera Hardware: {meta['camera_make']} {meta.get('camera_model', '')}). "
                    "GeoVision AI is calibrated exclusively for Earth observation satellite imagery."
                )
                return InputClassification.UNSUPPORTED_INPUT, meta
    except Exception:
        pass

    # 8. User-Declared Satellite Imagery (Unverified Mode)
    # Saturation or missing EXIF alone must not arbitrarily reject images as consumer photos
    # Where authenticity cannot be established, apply standard disclaimer.
    meta["unverified_disclaimer"] = "Satellite imagery was declared by the user but could not be independently verified."
    return InputClassification.USER_DECLARED_SATELLITE_UNVERIFIED, meta


def calculate_water_area_km2(
    water_pixels: int,
    classification: InputClassification,
    metadata: Dict[str, Any]
) -> Tuple[Optional[float], str]:
    """
    Calculate surface water area in square kilometers with rigorous scientific qualification.

    Returns:
        (area_km2, methodology_note)
    """
    if classification in (
        InputClassification.UNSUPPORTED_INPUT,
        InputClassification.UNSUPPORTED_PHOTOGRAPH,
        InputClassification.INVALID_IMAGE,
        InputClassification.INFERENCE_FAILED
    ):
        return None, "Calculations disabled: Input was rejected by the scientific safety engine."

    # 1. Verified Geospatial Mode (from embedded GeoTIFF tags)
    if classification == InputClassification.VERIFIED_GEOSPATIAL_INPUT and metadata.get("pixel_area_m2"):
        pixel_area_m2 = metadata["pixel_area_m2"]
        area_km2 = (water_pixels * pixel_area_m2) / 1e6
        scale_disp = f" ({metadata['pixel_scale_m'][0]:.2f}m × {metadata['pixel_scale_m'][1]:.2f}m)" if metadata.get("pixel_scale_m") else ""
        deg_note = " [Converted from geographic angular degrees]" if metadata.get("coordinates_in_degrees") else ""
        note = (
            f"Verified Geospatial Calculation: Pixel area {pixel_area_m2:.1f} m²/px{scale_disp}{deg_note} "
            f"derived directly from embedded GeoTIFF tags and coordinate reference system."
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
        "Nominal Approximation: Satellite imagery was declared by the user but could not be independently verified. "
        "Source imagery lacks verifiable geospatial coordinate reference or pixel scale tags. "
        "Calculated assuming a nominal Sentinel-2 10m GSD (100 m²/px). Not calibrated for surveying or legal boundaries."
    )
    return round(area_km2, 4), note
