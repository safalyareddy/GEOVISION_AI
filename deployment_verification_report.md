# GeoVision AI — Live Deployment & Production Recovery Verification Report

**Date**: 2026-11-06  
**Target Environment**: Public Cloud & Production Microservice  
**Live URL**: [https://ai-geovision.streamlit.app/](https://ai-geovision.streamlit.app/)  
**Git Upstream**: `https://github.com/safalyareddy/GEOVISION_AI.git` (`origin/main`)  
**Hugging Face Hub**: `safalya06/geovision-water-segmentation-models`  
**Authoritative Pinned Revision**: `441a99034b36336c34d86c23cfe9c1a8114ca3a4`

---

## 1. Exact Live Failures Diagnosed & Resolved

### Failure A — Obsolete Hugging Face Revision Overridden
- **Diagnosis**: The deployed application in certain cloud sessions or environment configs defaulted to obsolete revision `022039f0ce080afec9354a18635af97cc4cf352e` (or legacy `0ce5ce5824`). On Hugging Face Hub, revision `022039f` contains *only* `fpn_best.pth` and lacks the other three models.
- **Resolution**: Implemented `get_hf_revision()` in `src/utils/model_registry.py` with an immutable fallback. Configured `OBSOLETE_HF_REVISIONS` to intercept and override obsolete revisions with the authoritative revision `441a99034b36336c34d86c23cfe9c1a8114ca3a4`.

### Failure B — Three Checkpoint Verification & Download Failures
- **Diagnosis**: The deployed application could not locate or download `unet_plusplus_best.pth`, `deeplabv3_plus_best.pth`, and `segformer_best.pth` when pointing to the obsolete revision, triggering `EntryNotFoundError: 404 Client Error`.
- **Resolution**: Pinned the authoritative revision `441a99034b36336c34d86c23cfe9c1a8114ca3a4`, which contains all four model checkpoints in native Git LFS format. Verified that clean downloads of all four files match their authoritative SHA-256 hashes bit-for-bit.

### Failure C — Dashboard KeyError (`app.py` line 614)
- **Diagnosis**: `app.py` crashed at `status_map[k]['display_name']` when keys in `SUPPORTED_MODELS` were missing from the status dictionary or missing required fields.
- **Resolution**:
  1. `get_model_status_summary()` now iterates over `dict.fromkeys(list(SUPPORTED_MODELS.keys()) + list(MODEL_REGISTRY.keys()))`, guaranteeing every supported model has a complete status object.
  2. Implemented `get_safe_model_meta(status_map, model_key)` in `app.py` to guarantee non-empty metadata and eliminate `KeyError`.
  3. Added failure isolation so a failure in loading one model displays an informative diagnostic without crashing the comparison dashboard or preventing other models from operating.

### Failure D — Truthful Backend Availability Representation
- **Diagnosis**: The public Streamlit UI previously reported `FastAPI Microservice (Unavailable - Local Docker only)`.
- **Resolution**: Truthfully maintained this label for the public Streamlit app. PyTorch (In-Process) and ONNX Runtime (Optimized CPU) are the active, fully verified public production backends. FastAPI is documented as a standalone local/containerized microservice.

---

## 2. Four-Model Authoritative Checkpoints & Checksums

| Architecture | Model ID | Checkpoint File | Size | Expected & Verified SHA-256 Hash | Hub Integrity |
|---|---|---|---|---|---|
| **FPN (Champion)** | `fpn` | `fpn_best.pth` | 92.8 MB | `d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1` | **MATCH (100%)** |
| **UNet++** | `unet_plusplus` | `unet_plusplus_best.pth` | 104.5 MB | `497e246ef79d87f3b2255cc119e7ab8bd0045d48caf3d4710e9f6738e56c4548` | **MATCH (100%)** |
| **DeepLabV3+** | `deeplabv3_plus` | `deeplabv3_plus_best.pth` | 89.9 MB | `6c7c6dd4835d29be34b9eda03bdfca878914749a526c44cade99906cc1b1b827` | **MATCH (100%)** |
| **SegFormer** | `segformer` | `segformer_best.pth` | 14.9 MB | `2c762a52a5aa5cafb6aecb698bf483257fd5279bce3f3af7368850ba9354e986` | **MATCH (100%)** |

All four models load strictly with `model.load_state_dict(..., strict=True)` and `model.eval()`.

---

## 3. 12-Combination Backend Acceptance Summary

| Model | PyTorch In-Process | FastAPI REST API (Local) | ONNX Runtime (CPU) | Binary Mask Agreement |
|---|---|---|---|---|
| **FPN (Champion)** | PASS (25.7 ms, 12.63% cov) | PASS (133.3 ms, 12.63% cov) | PASS (32.1 ms, 12.63% cov) | **100.00%** |
| **UNet++** | PASS (91.6 ms, 6.93% cov) | PASS (202.2 ms, 6.93% cov) | PASS (75.1 ms, 6.93% cov) | **100.00%** |
| **DeepLabV3+** | PASS (38.3 ms, 10.10% cov) | PASS (137.8 ms, 10.10% cov) | PASS (32.7 ms, 10.10% cov) | **100.00%** |
| **SegFormer** | PASS (61.1 ms, 7.45% cov) | PASS (89.7 ms, 7.45% cov) | PASS (12.5 ms, 7.45% cov) | **100.00%** |

---

## 4. Scientific Input Safety Verification

All safety gates tested with 100% pass rate:
- **Project Sample Identification**: Curated test samples (e.g. `patch_0047.png`) matched by exact SHA-256 hash (`ae6b621438ba...`), never by filename alone.
- **Consumer Hardware Rejection**: Smartphone/camera EXIF tags (Apple, Canon, Nikon, Sony) rejected with descriptive error.
- **Extreme Floral Saturation**: Consumer floral macros rejected with HTTP 422 / UI error banner.
- **Blank Images**: Blank black and blank white images rejected as `UNSUPPORTED_INPUT`.
- **Geospatial Mode**: Valid CRS and pixel scale produce exact verified area; geographic angular degrees ($0.0001^\circ$) converted to metric scale rather than treated as metres; rotated pixels handled via affine determinant.
- **Unverified Disclaimer**: Genuine satellite imagery without metadata accepted with mandatory disclosure: *"Satellite imagery was declared by the user but could not be independently verified."*

---

## 5. Automated Test Suite Results

- **Command**: `./venv/bin/python -m pytest tests/ -v`
- **Result**: **139 passed, 1 warning in 56.79 seconds**.
- **Failures / Errors**: **0**.
