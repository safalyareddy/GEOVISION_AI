# GeoVision AI — Scientific Input Validation & Geospatial Quantification Report

**Purpose**: This report documents the design, implementation, and scientific validation of GeoVision AI's **Input Validation & Safety Protocol Pipeline** (`src/utils/input_validator.py`). The pipeline prevents the application from making unscientific hydrological claims on ordinary consumer photography and guarantees verifiable geospatial area quantification.

---

## 1. Six Safety Classification States

The input validation pipeline classifies every inbound payload into one of six mutually exclusive categories:

| State Code | Description | Action / Safety Gate | Area Quantification Mode |
|---|---|---|---|
| `VERIFIED_PROJECT_SAMPLE` | Payload matches curated Sentinel-2 test split patch strictly by cryptographic SHA-256 hash (never filename alone). | Allowed. Pre-computed ground truth mask displayed alongside prediction. | Nominal Sentinel-2 10m GSD (100.0 m²/px). |
| `VERIFIED_GEOSPATIAL_INPUT` | TIFF image containing embedded GeoTIFF tags (Tag 33550 `ModelPixelScaleTag` / Tag 34264 `ModelTransformationTag` and Tag 34735 `GeoKeyDirectoryTag`). | Allowed. Spatial bounds and coordinate reference extracted and verified. | **Verified Geospatial Mode**: Exact square kilometers derived from physical pixel scale ($Scale_X \times Scale_Y$) or affine Jacobian determinant for rotated pixels. |
| `USER_DECLARED_SATELLITE_UNVERIFIED` | Plausible natural color or false-color remote sensing image without embedded geospatial coordinate tags. | Allowed with explicit disclaimer: *"Satellite imagery was declared by the user but could not be independently verified."* | **Nominal Approximation Mode**: Sentinel-2 10m GSD assumed (100 m²/px) with explicit approximation notice. |
| `UNSUPPORTED_INPUT` / `UNSUPPORTED_PHOTOGRAPH` | Consumer photograph (flowers, portraits, landscapes, indoor objects, solid blanks) identified via EXIF hardware tags, blank black/white checks, or consumer floral saturation heuristics. | **Blocked**: In FastAPI, returns HTTP 422 Unprocessable Content. In Streamlit, disables hydrological claims. | **Calculations Disabled**: Returns `None`. |
| `INVALID_IMAGE` | Corrupted byte stream, zero-byte file, oversized (>10MB), invalid dimensions (<32x32 or >8192x8192), extreme aspect ratio (<0.2 or >5.0), or unsupported format. | **Blocked**: In FastAPI, returns HTTP 400 Bad Request. In Streamlit, displays invalid payload banner. | **Calculations Disabled**: Returns `None`. |
| `INFERENCE_FAILED` | Internal runtime exception during model forward pass or tensor transformation. | Returns HTTP 500 / 503 error with diagnostic stack trace. | **Calculations Disabled**: Returns `None`. |

---

## 2. Rejection & Provenance Verification Protocols

### A. Strict Hash-Based Sample Identification
- Filenames alone (such as `patch_0047.png`) are never trusted.
- Input payloads are checked against SHA-256 signatures of the curated benchmark test split (`ae6b621438ba...` for `patch_0047.png`).
- Tampered images or non-matching files bearing identical names are automatically classified as `USER_DECLARED_SATELLITE_UNVERIFIED`.

### B. Blank White and Blank Black Image Rejection
- Solid uniform images (e.g. blank black `mean < 2.0`, blank white `mean > 253.0`) with spatial standard deviation $< 0.5$ are rejected as `UNSUPPORTED_INPUT`.
- Prevents trivial blank inputs from generating spurious water coverage metrics.

### C. EXIF Hardware Metadata Inspection
- Header tags are scanned for consumer camera/smartphone manufacturer identifiers (e.g. Apple, Canon, Nikon, Sony, Samsung, Google, Xiaomi, Huawei, Olympus, Fujifilm, Leica, Motorola).
- When a consumer camera make is present, the image is identified as `UNSUPPORTED_INPUT` before forward pass execution.

### D. Extreme Floral Saturation & Consumer Photography
- Consumer photography (e.g. floral macros, vivid indoor objects) exhibiting extreme saturation ($>35\%$ of pixels with $S > 0.80$ and $P_{95} > 0.90$) combined with consumer indicators is safely rejected.
- Genuine satellite imagery with high saturation (e.g. false-color NIR composites, deep ocean lagoons, mineral mapping) without consumer EXIF is allowed under `USER_DECLARED_SATELLITE_UNVERIFIED` with the explicit unverified declaration.

### E. Server-Side Security & Payload Limits
- Enforces strict 10 MB payload size limit on the server before decoding.
- Enforces PIL decompression-bomb safety (`MAX_IMAGE_PIXELS = 100,000,000`).
- Validates MIME headers against true binary magic signatures (PNG, JPEG, TIFF).

---

## 3. Two-Mode Geospatial Area Quantification Protocol

Surface water area calculation enforces scientific integrity through two distinct modes:

### Mode 1: Verified Geospatial Mode
- **Trigger**: Embedded GeoTIFF metadata tags detected in input file with verified CRS and positive pixel scale.
- **Geographic Degrees vs. Metres**: If coordinates are in geographic degrees (e.g. $Scale < 0.05^\circ$), pixel dimensions are converted to metric ground distance based on latitude ($1^\circ \text{ lat} \approx 111,320\text{ m}$, $1^\circ \text{ lon} \approx 111,320 \times \cos(\text{lat})\text{ m}$), preventing catastrophic sub-millimeter area miscalculations.
- **Rotated Pixels**: If `ModelTransformationTag` (34264) is present, pixel area is derived directly from the Jacobian determinant of the 2D affine transformation: $|a \cdot d - b \cdot c|$.
- **Formulation**:
  $$\text{Area (km}^2\text{)} = \frac{N_{\text{water}} \times \text{PixelArea}_{\text{m}^2}}{1,000,000}$$
- **Output Note**: Includes explicit scale dimensions, CRS datum verification, and degree-conversion notes.

### Mode 2: Nominal Approximation Mode
- **Trigger**: Non-GeoTIFF optical imagery (e.g., standard PNG/JPEG tiles).
- **Formulation**:
  $$\text{Area (km}^2\text{)} = \frac{N_{\text{water}} \times 100.0\text{ m}^2}{1,000,000}$$
- **Output Note**: Explicitly states: *"Nominal Approximation: Satellite imagery was declared by the user but could not be independently verified. Source imagery lacks verifiable geospatial coordinate reference or pixel scale tags. Calculated assuming a nominal Sentinel-2 10m GSD (100 m²/px). Not calibrated for surveying or legal boundaries."*

---

## 4. Test Verification Summary

The input validation pipeline has been verified with 100% pass rate in `tests/test_input_validator.py` (13/13 passed) and `tests/test_api.py` (15/15 passed):
1. `test_verified_project_sample`: Passed (`patch_0047.png` correctly recognized by SHA-256).
2. `test_project_sample_filename_alone_not_trusted`: Passed (Non-matching bytes named `patch_0047.png` rejected as sample).
3. `test_unsupported_photograph_exif`: Passed (Apple iPhone EXIF tags rejected with HTTP 422).
4. `test_blank_white_and_blank_black_image`: Passed (Blank black and blank white images rejected).
5. `test_corrupted_image_and_zero_byte`: Passed (Zero-byte and corrupted payloads rejected as invalid).
6. `test_oversized_upload_and_invalid_extension`: Passed (>10MB and `.exe` rejected as invalid).
7. `test_mime_content_mismatch`: Passed (Disguised text files rejected as invalid).
8. `test_invalid_dimensions_and_aspect_ratio`: Passed (Sub-32px and extreme aspect ratios rejected).
9. `test_verified_geospatial_input_with_projected_crs`: Passed (GeoTIFF pixel scale extracted, exact area computed).
10. `test_geotiff_missing_crs_falls_back_to_unverified`: Passed (Missing CRS prevents verified geospatial claim).
11. `test_geotiff_geographic_degrees_converted_to_metres`: Passed (Angular degrees converted to metric resolution).
12. `test_geotiff_affine_transformation_rotated_pixels`: Passed (Rotated pixels Jacobian determinant area verified).
13. `test_genuine_satellite_with_unusual_colors_or_no_exif`: Passed (False-color satellite accepted with unverified disclaimer).
