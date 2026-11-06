"""
GeoVision AI — Comprehensive Input Validation Unit Tests
Validates scientific safety protocols, 6-state classification, payload security,
rejection criteria, and 2-mode geospatial area quantification.
"""

import io
import math
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.utils.input_validator import (
    classify_input_image,
    calculate_water_area_km2,
    validate_raw_payload,
    InputClassification,
    MAX_UPLOAD_BYTES,
)


def test_verified_project_sample():
    """Verify that curated dataset test split patches are identified by cryptographic hash."""
    sample_path = Path("dataset/test/images/patch_0047.png")
    if not sample_path.exists():
        sample_path = Path("assets/demo/patch_0047.png")
    if not sample_path.exists():
        pytest.skip("patch_0047.png not found in dataset/test or assets/demo")

    raw_bytes = sample_path.read_bytes()
    img = Image.open(io.BytesIO(raw_bytes))

    classification, meta = classify_input_image(img, raw_bytes=raw_bytes, filename="arbitrary_name.png")
    assert classification == InputClassification.VERIFIED_PROJECT_SAMPLE
    assert meta.get("project_sample_name") == "patch_0047.png"

    area_km2, note = calculate_water_area_km2(1000, classification, meta)
    assert area_km2 == 0.1
    assert "Verified Project Sample" in note


def test_project_sample_filename_alone_not_trusted():
    """Verify that an arbitrary image named 'patch_0047.png' without the correct hash is NOT verified as a sample."""
    # Synthetic random non-matching image
    arr = np.random.randint(40, 100, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    classification, meta = classify_input_image(img, raw_bytes=raw_bytes, filename="patch_0047.png")
    assert classification != InputClassification.VERIFIED_PROJECT_SAMPLE
    assert classification == InputClassification.USER_DECLARED_SATELLITE_UNVERIFIED
    assert meta.get("unverified_disclaimer") is not None


def test_unsupported_photograph_exif():
    """Verify that images with consumer smartphone/camera EXIF tags are rejected."""
    img = Image.new("RGB", (64, 64), color=(80, 90, 85))
    exif_obj = img.getexif()
    exif_obj[271] = "Apple"  # Camera Make
    exif_obj[272] = "iPhone 15 Pro"  # Camera Model
    img.getexif = lambda: exif_obj

    classification, meta = classify_input_image(img, filename="iphone_photo.jpg")
    assert classification == InputClassification.UNSUPPORTED_INPUT
    assert "Consumer photograph detected" in meta["rejection_reason"]
    assert meta["camera_make"] == "Apple"

    area_km2, note = calculate_water_area_km2(1000, classification, meta)
    assert area_km2 is None
    assert "Calculations disabled" in note


def test_blank_white_and_blank_black_image():
    """Verify that solid uniform images (blank black, blank white) are rejected as UNSUPPORTED_INPUT."""
    # Blank Black
    black_img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    cls_black, meta_black = classify_input_image(black_img, filename="blank_black.png")
    assert cls_black == InputClassification.UNSUPPORTED_INPUT
    assert "Blank or uniform image detected" in meta_black["rejection_reason"]

    # Blank White
    white_img = Image.new("RGB", (64, 64), color=(255, 255, 255))
    cls_white, meta_white = classify_input_image(white_img, filename="blank_white.png")
    assert cls_white == InputClassification.UNSUPPORTED_INPUT
    assert "Blank or uniform image detected" in meta_white["rejection_reason"]


def test_corrupted_image_and_zero_byte():
    """Verify that 0-byte payloads and corrupted file bytes are rejected as INVALID_IMAGE."""
    # Zero-byte payload
    cls_zero, meta_zero = classify_input_image(raw_bytes=b"", filename="empty.png")
    assert cls_zero == InputClassification.INVALID_IMAGE
    assert "Zero-byte" in meta_zero["rejection_reason"]

    # Corrupted bytes disguised as PNG
    corrupt_bytes = b"\x89PNG\r\n\x1a\n" + b"garbage_non_png_binary_data"
    cls_corrupt, meta_corrupt = classify_input_image(raw_bytes=corrupt_bytes, filename="corrupt.png")
    assert cls_corrupt == InputClassification.INVALID_IMAGE
    assert "Corrupted or unreadable" in meta_corrupt["rejection_reason"]


def test_oversized_upload_and_invalid_extension():
    """Verify that oversized uploads (>10MB) and unsupported extensions are rejected."""
    # Oversized payload
    large_payload = b"\x89PNG\r\n\x1a\n" + b"0" * (MAX_UPLOAD_BYTES + 1024)
    cls_large, meta_large = classify_input_image(raw_bytes=large_payload, filename="large.png")
    assert cls_large == InputClassification.INVALID_IMAGE
    assert "Oversized upload" in meta_large["rejection_reason"]

    # Unsupported extension
    dummy_img = Image.new("RGB", (64, 64), color=(60, 70, 65))
    cls_ext, meta_ext = classify_input_image(dummy_img, filename="malicious.exe")
    assert cls_ext == InputClassification.INVALID_IMAGE
    assert "Unsupported file extension" in meta_ext["rejection_reason"]


def test_mime_content_mismatch():
    """Verify that text or executables disguised as image files are rejected."""
    text_disguised = b"This is a plain text file, not a satellite image."
    cls_mime, meta_mime = classify_input_image(raw_bytes=text_disguised, filename="fake_image.png")
    assert cls_mime == InputClassification.INVALID_IMAGE
    assert "Invalid MIME" in meta_mime["rejection_reason"]


def test_invalid_dimensions_and_aspect_ratio():
    """Verify that images that are too small (<32), too large (>8192), or have extreme aspect ratios are rejected."""
    # Below minimum dimensions (16x16)
    img_tiny = Image.new("RGB", (16, 16), color=(60, 75, 70))
    cls_tiny, meta_tiny = classify_input_image(img_tiny)
    assert cls_tiny == InputClassification.INVALID_IMAGE
    assert "below minimum" in meta_tiny["rejection_reason"]

    # Extremely wide aspect ratio (600x50 = 12:1)
    img_wide = Image.new("RGB", (600, 50), color=(60, 75, 70))
    cls_wide, meta_wide = classify_input_image(img_wide)
    assert cls_wide == InputClassification.INVALID_IMAGE
    assert "Extreme aspect ratio" in meta_wide["rejection_reason"]

    # Extremely tall aspect ratio (50x600 = 0.083:1)
    img_tall = Image.new("RGB", (50, 600), color=(60, 75, 70))
    cls_tall, meta_tall = classify_input_image(img_tall)
    assert cls_tall == InputClassification.INVALID_IMAGE
    assert "Extreme aspect ratio" in meta_tall["rejection_reason"]


def test_verified_geospatial_input_with_projected_crs():
    """Verify that GeoTIFF with valid CRS and metric pixel scale produces VERIFIED_GEOSPATIAL_INPUT."""
    img = Image.new("RGB", (100, 100), color=(50, 60, 55))
    # Mock tag_v2 with ModelPixelScaleTag (10m x 10m) and valid GeoKeyDirectory
    img.tag_v2 = {
        33550: (10.0, 10.0, 0.0),
        34735: (1, 1, 0, 1, 1024, 0, 1, 1),  # GeoKey directory header
    }

    classification, meta = classify_input_image(img, filename="sentinel2_geotiff.tif")
    assert classification == InputClassification.VERIFIED_GEOSPATIAL_INPUT
    assert meta["has_geotiff_tags"] is True
    assert meta["has_crs"] is True
    assert meta["pixel_scale_m"] == (10.0, 10.0)

    area_km2, note = calculate_water_area_km2(5000, classification, meta)
    assert area_km2 == 0.5
    assert "Verified Geospatial Calculation" in note


def test_geotiff_missing_crs_falls_back_to_unverified():
    """Verify that a GeoTIFF with pixel scale but MISSING CRS cannot claim verified geospatial precision."""
    img = Image.new("RGB", (100, 100), color=(50, 60, 55))
    # Has pixel scale 33550, but NO GeoKey 34735
    img.tag_v2 = {33550: (10.0, 10.0, 0.0)}

    classification, meta = classify_input_image(img, filename="missing_crs.tif")
    assert classification == InputClassification.USER_DECLARED_SATELLITE_UNVERIFIED
    assert meta["has_crs"] is False

    area_km2, note = calculate_water_area_km2(2000, classification, meta)
    assert "Nominal Approximation" in note


def test_geotiff_geographic_degrees_converted_to_metres():
    """Verify that GeoTIFF with geographic angular degrees is derived as metric scale, not treated as metres."""
    img = Image.new("RGB", (100, 100), color=(50, 60, 55))
    # 0.0001 degrees pixel scale (approx 11.13 meters)
    img.tag_v2 = {
        33550: (0.0001, 0.0001, 0.0),
        33922: (0, 0, 0, 10.0, 45.0, 0.0),  # Latitude 45 deg
        34735: (1, 1, 0, 1, 1024, 0, 1, 2),  # GeoKey
    }

    classification, meta = classify_input_image(img, filename="wgs84_degrees.tif")
    assert classification == InputClassification.VERIFIED_GEOSPATIAL_INPUT
    assert meta["coordinates_in_degrees"] is True
    # At latitude 45 deg, scale_x_m ~ 0.0001 * 111320 * cos(45 deg) ~ 7.87m; scale_y_m ~ 11.13m
    assert meta["pixel_scale_m"][0] > 1.0  # Must NOT be 0.0001 metres!
    assert meta["pixel_scale_m"][1] > 1.0

    area_km2, note = calculate_water_area_km2(10000, classification, meta)
    assert area_km2 is not None
    assert "Converted from geographic angular degrees" in note


def test_geotiff_affine_transformation_rotated_pixels():
    """Verify that rotated pixels specified via ModelTransformationTag (34264) derive exact area from Jacobian determinant."""
    img = Image.new("RGB", (100, 100), color=(50, 60, 55))
    # 45-degree rotation matrix with 10m scale:
    # m00 = 10 * cos(45), m01 = -10 * sin(45), m10 = 10 * sin(45), m11 = 10 * cos(45)
    # det = 10*10 = 100 m^2
    c45 = 10.0 * math.cos(math.radians(45))
    s45 = 10.0 * math.sin(math.radians(45))
    transform_matrix = [
        c45, -s45, 0.0, 500000.0,
        s45, c45, 0.0, 4000000.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ]
    img.tag_v2 = {
        34264: transform_matrix,
        34735: (1, 1, 0, 1, 1024, 0, 1, 1),
    }

    classification, meta = classify_input_image(img, filename="rotated_sentinel2.tif")
    assert classification == InputClassification.VERIFIED_GEOSPATIAL_INPUT
    assert round(meta["pixel_area_m2"], 1) == 100.0

    area_km2, note = calculate_water_area_km2(1000, classification, meta)
    assert area_km2 == 0.1
    assert "Verified Geospatial Calculation" in note


def test_genuine_satellite_with_unusual_colors_or_no_exif():
    """
    Verify that genuine satellite imagery without EXIF or with unusual colors (e.g. false color NIR)
    is NOT incorrectly rejected, but accepted under USER_DECLARED_SATELLITE_UNVERIFIED with disclaimer.
    """
    # High saturation / unusual color composite (e.g. vivid cyan water or false-color NIR red vegetation)
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:50, :, 0] = 220  # False-color NIR vegetation
    arr[50:, :, 2] = 200  # Deep water body
    img = Image.fromarray(arr)

    classification, meta = classify_input_image(img, filename="false_color_sentinel.png")
    assert classification == InputClassification.USER_DECLARED_SATELLITE_UNVERIFIED
    assert "Satellite imagery was declared by the user but could not be independently verified." in meta["unverified_disclaimer"]

    # Nominal area approximation must still function
    area_km2, note = calculate_water_area_km2(2500, classification, meta)
    assert area_km2 == 0.25
    assert "Nominal Approximation" in note
