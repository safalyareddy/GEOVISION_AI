"""
GeoVision AI — Detailed Geospatial Area Calculation Inspection
Inspects real projected, geographic, rotated, and missing-CRS GeoTIFF samples.
"""

import io
import sys
import math
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.input_validator import (
    classify_input_image,
    calculate_water_area_km2,
    InputClassification,
)


def inspect_geospatial_engine():
    print("=" * 80)
    print("GeoVision AI — Geospatial Area Calculation & GeoTIFF Validation Inspection")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Case A: Real Projected GeoTIFF (Metric 10m x 10m UTM/Projected CRS)
    # -------------------------------------------------------------------------
    print("\n--- CASE A: Projected GeoTIFF (Metric Pixel Scale & Valid CRS) ---")
    img_proj = Image.new("RGB", (256, 256), color=(40, 50, 45))
    # Tag 33550: ModelPixelScaleTag = (10.0m, 10.0m, 0.0)
    # Tag 34735: GeoKeyDirectoryTag header indicating projected coordinate system
    img_proj.tag_v2 = {
        33550: (10.0, 10.0, 0.0),
        34735: (1, 1, 0, 1, 1024, 0, 1, 1),
    }
    cls_proj, meta_proj = classify_input_image(img_proj, filename="sentinel2_utm32n.tif")
    print(f"Classification: {cls_proj.value}")
    print(f"Detected CRS: {meta_proj.get('crs_info')} (has_crs={meta_proj.get('has_crs')})")
    print(f"Pixel Scale:  {meta_proj.get('pixel_scale_m')} metres")
    print(f"Pixel Area:   {meta_proj.get('pixel_area_m2'):.2f} m²/px")

    # Evaluate 10,000 water pixels
    water_px = 10000
    area_proj, note_proj = calculate_water_area_km2(water_px, cls_proj, meta_proj)
    expected_proj_km2 = (water_px * 100.0) / 1e6  # 1.0 km²
    print(f"Computed Water Area ({water_px:,} pixels): {area_proj:.4f} km² (Expected: {expected_proj_km2:.4f} km²)")
    print(f"Methodology Note: {note_proj}")
    assert cls_proj == InputClassification.VERIFIED_GEOSPATIAL_INPUT
    assert area_proj == expected_proj_km2

    # -------------------------------------------------------------------------
    # Case B: Geographic GeoTIFF (WGS-84 Angular Degrees with Latitude Tiepoint)
    # -------------------------------------------------------------------------
    print("\n--- CASE B: Geographic GeoTIFF (Angular Degrees with Latitude Cosine Correction) ---")
    img_geo = Image.new("RGB", (256, 256), color=(40, 50, 45))
    lat_deg = 45.0  # Latitude 45 deg N
    scale_deg = 0.0001  # 0.0001 degrees
    # Tag 33550: ModelPixelScaleTag = (0.0001 deg, 0.0001 deg, 0.0)
    # Tag 33922: ModelTiepointTag = (I, J, K, X, Y, Z) = (0, 0, 0, 10.0, 45.0, 0.0)
    # Tag 34735: GeoKeyDirectoryTag header
    img_geo.tag_v2 = {
        33550: (scale_deg, scale_deg, 0.0),
        33922: (0, 0, 0, 10.0, lat_deg, 0.0),
        34735: (1, 1, 0, 1, 1024, 0, 1, 2),
    }
    cls_geo, meta_geo = classify_input_image(img_geo, filename="wgs84_geographic.tif")
    print(f"Classification: {cls_geo.value}")
    print(f"Coordinates in Degrees: {meta_geo.get('coordinates_in_degrees')}")
    print(f"Latitude Tiepoint: {lat_deg}° N")
    print(f"Converted Metric Scale: {meta_geo.get('pixel_scale_m')} metres")
    print(f"Converted Pixel Area:   {meta_geo.get('pixel_area_m2'):.2f} m²/px")

    # Theoretical conversion:
    # 1 deg lat = 111,320 m -> scale_y = 0.0001 * 111,320 = 11.132 m
    # 1 deg lon = 111,320 * cos(45 deg) = 78,715 m -> scale_x = 0.0001 * 78,715 = 7.8715 m
    # pixel_area = 11.132 * 7.8715 = 87.625 m^2
    theoretical_area_m2 = (scale_deg * 111320.0) * (scale_deg * 111320.0 * math.cos(math.radians(lat_deg)))
    expected_geo_km2 = round((water_px * theoretical_area_m2) / 1e6, 4)

    area_geo, note_geo = calculate_water_area_km2(water_px, cls_geo, meta_geo)
    print(f"Computed Water Area ({water_px:,} pixels): {area_geo:.4f} km² (Theoretical: {expected_geo_km2:.4f} km²)")
    print(f"Methodology Note: {note_geo}")
    assert cls_geo == InputClassification.VERIFIED_GEOSPATIAL_INPUT
    assert area_geo == expected_geo_km2

    # -------------------------------------------------------------------------
    # Case C: Rotated Pixels via ModelTransformationTag (Affine Jacobian Determinant)
    # -------------------------------------------------------------------------
    print("\n--- CASE C: Rotated Pixels GeoTIFF (Affine Jacobian Determinant) ---")
    img_rot = Image.new("RGB", (256, 256), color=(40, 50, 45))
    # 30-degree rotation with 12m pixel ground resolution:
    # det = 12 * 12 = 144 m^2
    deg_rot = 30.0
    c30 = 12.0 * math.cos(math.radians(deg_rot))
    s30 = 12.0 * math.sin(math.radians(deg_rot))
    transform_matrix = [
        c30, -s30, 0.0, 500000.0,
        s30,  c30, 0.0, 4000000.0,
        0.0,  0.0, 1.0, 0.0,
        0.0,  0.0, 0.0, 1.0,
    ]
    img_rot.tag_v2 = {
        34264: transform_matrix,
        34735: (1, 1, 0, 1, 1024, 0, 1, 1),
    }
    cls_rot, meta_rot = classify_input_image(img_rot, filename="rotated_orbit_tile.tif")
    print(f"Classification: {cls_rot.value}")
    print(f"Rotation Angle: {deg_rot}°")
    print(f"Jacobian Determinant (Pixel Area): {meta_rot.get('pixel_area_m2'):.2f} m²/px")
    area_rot, note_rot = calculate_water_area_km2(water_px, cls_rot, meta_rot)
    expected_rot_km2 = round((water_px * 144.0) / 1e6, 4)
    print(f"Computed Water Area ({water_px:,} pixels): {area_rot:.4f} km² (Expected: {expected_rot_km2:.4f} km²)")
    print(f"Methodology Note: {note_rot}")
    assert cls_rot == InputClassification.VERIFIED_GEOSPATIAL_INPUT
    assert area_rot == expected_rot_km2

    # -------------------------------------------------------------------------
    # Case D: GeoTIFF Missing CRS (Fallback to Unverified Mode)
    # -------------------------------------------------------------------------
    print("\n--- CASE D: GeoTIFF Missing CRS (Safe Fallback to Unverified Nominal Mode) ---")
    img_nocrs = Image.new("RGB", (256, 256), color=(40, 50, 45))
    # Has pixel scale 33550, but NO GeoKey 34735
    img_nocrs.tag_v2 = {33550: (10.0, 10.0, 0.0)}
    cls_nocrs, meta_nocrs = classify_input_image(img_nocrs, filename="missing_crs_tile.tif")
    print(f"Classification: {cls_nocrs.value}")
    print(f"Has CRS: {meta_nocrs.get('has_crs')}")
    area_nocrs, note_nocrs = calculate_water_area_km2(water_px, cls_nocrs, meta_nocrs)
    print(f"Computed Water Area: {area_nocrs:.4f} km² (Nominal 10m GSD default)")
    print(f"Methodology Note: {note_nocrs}")
    assert cls_nocrs == InputClassification.USER_DECLARED_SATELLITE_UNVERIFIED
    assert "Nominal Approximation" in note_nocrs

    print("\n>>> ALL 4 GEOSPATIAL CASES MATHEMATICALLY AND LOGICALLY VERIFIED <<<")


if __name__ == "__main__":
    inspect_geospatial_engine()
