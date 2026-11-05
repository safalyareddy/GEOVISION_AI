"""
GeoVision AI — Input Validator Unit Tests
Validates scientific safety protocols, 6-state classification, and 2-mode geospatial area quantification.
"""

import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.utils.input_validator import (
    classify_input_image,
    calculate_water_area_km2,
    InputClassification,
)


def test_verified_project_sample():
    """Verify that curated dataset test split patches are identified as VERIFIED_PROJECT_SAMPLE."""
    sample_path = Path("dataset/test/images/patch_0047.png")
    if not sample_path.exists():
        pytest.skip("dataset/test/images/patch_0047.png not found")

    raw_bytes = sample_path.read_bytes()
    img = Image.open(io.BytesIO(raw_bytes))

    classification, meta = classify_input_image(img, raw_bytes=raw_bytes, filename="patch_0047.png")
    assert classification == InputClassification.VERIFIED_PROJECT_SAMPLE
    assert meta.get("project_sample_name") == "patch_0047.png"

    area_km2, note = calculate_water_area_km2(1000, classification, meta)
    assert area_km2 == 0.1
    assert "Verified Project Sample" in note


def test_unsupported_photograph_saturation():
    """Verify that vivid consumer photography (e.g. flowers/portraits) is rejected as UNSUPPORTED_PHOTOGRAPH."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[..., 0] = 255
    arr[..., 1] = 10
    arr[..., 2] = 20
    img = Image.fromarray(arr)

    classification, meta = classify_input_image(img, filename="flower_macro.jpg")
    assert classification == InputClassification.UNSUPPORTED_PHOTOGRAPH
    assert "Unsupported photograph" in meta["rejection_reason"]

    area_km2, note = calculate_water_area_km2(1000, classification, meta)
    assert area_km2 is None
    assert "Calculations disabled" in note


def test_unsupported_photograph_exif():
    """Verify that images with consumer smartphone/camera EXIF tags are rejected."""
    img = Image.new("RGB", (64, 64), color=(80, 90, 85))
    exif_obj = img.getexif()
    exif_obj[271] = "Apple"  # Camera Make
    exif_obj[272] = "iPhone 15 Pro"  # Camera Model

    # Mock getexif on the image instance
    img.getexif = lambda: exif_obj

    classification, meta = classify_input_image(img, filename="iphone_photo.jpg")
    assert classification == InputClassification.UNSUPPORTED_PHOTOGRAPH
    assert "Consumer photograph detected" in meta["rejection_reason"]
    assert meta["camera_make"] == "Apple"


def test_verified_geospatial_input_with_tags():
    """Verify that GeoTIFF tags enable VERIFIED_GEOSPATIAL_INPUT and exact area quantification."""
    img = Image.new("RGB", (100, 100), color=(50, 60, 55))
    # Mock tag_v2 with ModelPixelScaleTag (33550) = (10.0, 10.0, 0.0)
    img.tag_v2 = {33550: (10.0, 10.0, 0.0), 34735: (1, 1, 0, 1)}

    classification, meta = classify_input_image(img, filename="sentinel2_geotiff.tif")
    assert classification == InputClassification.VERIFIED_GEOSPATIAL_INPUT
    assert meta["has_geotiff_tags"] is True
    assert meta["pixel_scale_m"] == (10.0, 10.0)

    area_km2, note = calculate_water_area_km2(5000, classification, meta)
    assert area_km2 == 0.5
    assert "Verified Geospatial Calculation" in note


def test_user_declared_satellite_unverified():
    """Verify that unverified satellite imagery uses Nominal Approximation Mode."""
    img = Image.new("RGB", (100, 100), color=(60, 75, 70))
    classification, meta = classify_input_image(img, filename="custom_satellite.png")
    assert classification == InputClassification.USER_DECLARED_SATELLITE_UNVERIFIED

    area_km2, note = calculate_water_area_km2(2000, classification, meta)
    assert area_km2 == 0.2
    assert "Nominal Approximation" in note


def test_invalid_image_dimensions():
    """Verify that invalid dimensions or extreme aspect ratios are rejected."""
    # Too small
    img_tiny = Image.new("RGB", (16, 16), color=(60, 75, 70))
    cls_tiny, meta_tiny = classify_input_image(img_tiny)
    assert cls_tiny == InputClassification.INVALID_IMAGE
    assert "below minimum" in meta_tiny["rejection_reason"]

    # Extreme aspect ratio
    img_aspect = Image.new("RGB", (600, 50), color=(60, 75, 70))
    cls_aspect, meta_aspect = classify_input_image(img_aspect)
    assert cls_aspect == InputClassification.INVALID_IMAGE
    assert "Extreme aspect ratio" in meta_aspect["rejection_reason"]
