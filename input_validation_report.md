# GeoVision AI — Scientific Input Validation & Geospatial Quantification Report

**Purpose**: This report documents the design, implementation, and scientific validation of GeoVision AI's **Input Validation & Safety Protocol Pipeline** (`src/utils/input_validator.py`). The pipeline prevents the application from making unscientific hydrological claims on ordinary consumer photography and guarantees verifiable geospatial area quantification.

---

## 1. Six Safety Classification States

The input validation pipeline classifies every inbound payload into one of six mutually exclusive categories:

| State Code | Description | Action / Safety Gate | Area Quantification Mode |
|---|---|---|---|
| `VERIFIED_PROJECT_SAMPLE` | Payload matches curated Sentinel-2 test split patch (by SHA-256 or filename). | Allowed. Pre-computed ground truth mask displayed alongside prediction. | Nominal Sentinel-2 10m GSD (100.0 m²/px). |
| `VERIFIED_GEOSPATIAL_INPUT` | TIFF image containing embedded GeoTIFF tags (Tag 33550 `ModelPixelScaleTag` / Tag 34735 `GeoKeyDirectoryTag`). | Allowed. Spatial bounds and coordinate reference extracted. | **Verified Geospatial Mode**: Exact square kilometers derived from physical pixel scale ($Scale_X \times Scale_Y$). |
| `USER_DECLARED_SATELLITE_UNVERIFIED` | Plausible natural color remote sensing image without embedded geospatial coordinate tags. | Allowed with educational disclaimer. | **Nominal Approximation Mode**: Sentinel-2 10m GSD assumed (100 m²/px) with explicit approximation notice. |
| `UNSUPPORTED_PHOTOGRAPH` | Consumer photograph (flowers, portraits, landscapes, indoor objects) identified via EXIF tags or chromatic saturation heuristics. | **Blocked**: In FastAPI, returns HTTP 422 Unprocessable Content. In Streamlit, disables hydrological claims. | **Calculations Disabled**: Returns `None`. |
| `INVALID_IMAGE` | Corrupted byte stream, unreadable payload, invalid dimensions (<32x32 or >8192x8192), or extreme aspect ratio (<0.2 or >5.0). | **Blocked**: In FastAPI, returns HTTP 400 Bad Request. In Streamlit, displays invalid payload banner. | **Calculations Disabled**: Returns `None`. |
| `INFERENCE_FAILED` | Internal runtime exception during model forward pass or tensor transformation. | Returns HTTP 500 / 503 error with diagnostic stack trace. | **Calculations Disabled**: Returns `None`. |

---

## 2. Consumer Photography Detection Heuristics

Optical Earth observation satellites (e.g. Sentinel-2 MSI) measure top-of-atmosphere or surface reflectance, which is smoothly distributed across multispectral bands. Consumer photography (e.g., flowers, macro shots, portraits, indoor lighting) exhibits radically different photometric distributions:

### A. Chromatic Saturation Disparity
- Input RGB image is transformed to the HSV color space:
  $$S = \frac{\max(R, G, B) - \min(R, G, B)}{\max(R, G, B) + \epsilon}$$
- **Heuristic Threshold**: If greater than 35% of pixels exhibit saturation $S > 0.80$ AND the 95th percentile saturation $P_{95} > 0.90$, the image is classified as `UNSUPPORTED_PHOTOGRAPH`.
- **Floral / Vivid Disparity**: Rejects bright red roses, sunflowers, vibrant macro photography, and studio portraits while allowing genuine open water bodies, rivers, and coastal wetlands.

### B. EXIF Hardware Metadata Inspection
- Header tags are scanned for consumer camera/smartphone manufacturer identifiers (e.g. Apple, Canon, Nikon, Sony, Samsung, Google, Xiaomi, Huawei, Olympus, Fujifilm).
- When a consumer camera make is present, the image is identified as `UNSUPPORTED_PHOTOGRAPH` before forward pass execution.

---

## 3. Two-Mode Geospatial Area Quantification Protocol

Surface water area calculation enforces scientific integrity through two distinct modes:

### Mode 1: Verified Geospatial Mode
- **Trigger**: Embedded GeoTIFF metadata tags detected in input file.
- **Formulation**:
  $$\text{Area (km}^2\text{)} = \frac{N_{\text{water}} \times |Scale_X \times Scale_Y|}{1,000,000}$$
- **Output Note**: Includes explicit scale dimensions and datum verification.

### Mode 2: Nominal Approximation Mode
- **Trigger**: Non-GeoTIFF optical imagery (e.g., standard PNG/JPEG tiles).
- **Formulation**:
  $$\text{Area (km}^2\text{)} = \frac{N_{\text{water}} \times 100.0\text{ m}^2}{1,000,000}$$
- **Output Note**: Explicitly states that the 10m Ground Sample Distance ($100\text{ m}^2/\text{pixel}$) is an unreferenced nominal assumption not calibrated for legal or cadastral survey boundaries.

---

## 4. Test Verification Summary

The input validation pipeline has been verified with 100% pass rate in `tests/test_input_validator.py` and `tests/test_api.py`:
- `test_verified_project_sample`: Passed (`patch_0047.png` correctly recognized).
- `test_unsupported_photograph_saturation`: Passed (Pure saturated red rejected with HTTP 422).
- `test_unsupported_photograph_exif`: Passed (Apple iPhone EXIF tags rejected with HTTP 422).
- `test_verified_geospatial_input_with_tags`: Passed (GeoTIFF pixel scale extracted, exact area computed).
- `test_user_declared_satellite_unverified`: Passed (Nominal mode caveats applied).
- `test_invalid_image_dimensions`: Passed (Sub-32px and extreme aspect ratios rejected).
