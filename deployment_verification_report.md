# GeoVision AI — Live Deployment & Operational Verification Report

**Date**: 2026-11-05  
**Target Environment**: Public Cloud & Production Microservice  
**Live URL**: [https://ai-geovision.streamlit.app/](https://ai-geovision.streamlit.app/)  
**Git Upstream**: `https://github.com/safalyareddy/GEOVISION_AI.git` (`origin/main`)  
**Hugging Face Hub**: `safalya06/geovision-water-segmentation-models` (`441a99034b36336c34d86c23cfe9c1a8114ca3a4`)

---

## 1. Deployment Architecture Overview

```
                                      +---------------------------------------------+
                                      |             Hugging Face Hub                |
                                      |  safalya06/geovision-water-segmentation...   |
                                      |     (Pinned Commit: 441a99034b36...)        |
                                      +---------------------------------------------+
                                                       | (Streaming Download &
                                                       |  SHA-256 Verification)
                                                       v
+-------------------------------+      +---------------------------------------------+
|        Client Browser         | ---> |       Streamlit Cloud Application           |
| (Streamlit Community Cloud)   |      |        https://ai-geovision.streamlit.app/  |
+-------------------------------+      |                                             |
                                       |  - PyTorch In-Process Engine (CPU)          |
                                       |  - ONNX Runtime Execution Provider (CPU)    |
                                       |  - Scientific Input Safety Gate (6-State)   |
                                       |  - Multi-Architecture Comparison Dashboard  |
                                       +---------------------------------------------+
                                                       |
                                                       | (REST JSON / Multipart)
                                                       v
                                       +---------------------------------------------+
                                       |        FastAPI REST Microservice            |
                                       |   Containerized Docker Microservice (:8000) |
                                       |   GET /health | GET /ready | POST /predict  |
                                       +---------------------------------------------+
```

---

## 2. Component Verification Status

### A. Streamlit Cloud Dashboard (`https://ai-geovision.streamlit.app/`)
- **Status**: Operational and accessible.
- **Default Backend**: `PyTorch (In-Process) [Primary / Cloud Default]`.
- **Default Champion**: `FPN-ResNet34` (Micro Dice 0.6323).
- **Secondary Backends**:
  - `ONNX Runtime (Accelerated CPU)`: Standalone Open Neural Network Exchange engine.
  - `FastAPI Microservice (REST API)`: Auto-detected via `$API_URL` or localhost:8000.
- **Model Checkpoint Source**: Hugging Face Hub (`safalya06/geovision-water-segmentation-models`) pinned to commit `441a99034b36336c34d86c23cfe9c1a8114ca3a4`.
- **Checkpoint Verification**: All 4 models verify against streaming SHA-256 before instantiation.
- **Scientific Safety Gate**: Blocks consumer photographs (e.g. flowers, portraits) and returns descriptive rejection reasons.

### B. FastAPI Inference Microservice (`src/api/main.py`)
- **Root Endpoint (`GET /`)**: Returns service metadata and champion identifier.
- **Health Endpoint (`GET /health`)**: Reports `status: ok`, `champion_ready: true`, and active loaded models.
- **Readiness Endpoint (`GET /ready`)**: Truthful readiness reporting without phantom instantiation.
- **Status Endpoint (`GET /models/{model_id}/status`)**: Returns individual architecture status, metadata, and memory caching state.
- **Metrics Endpoint (`GET /metrics`)**: Tracks request counts, rejected photos, errors, and per-model traffic.
- **Predict Endpoint (`POST /predict`)**:
  - Accepts multipart image files (PNG, JPG, TIFF) with 10 MB streaming size enforcement.
  - Classifies input via `classify_input_image`.
  - Rejects consumer photography with **HTTP 422 Unprocessable Content**.
  - Rejects corrupted files with **HTTP 400 Bad Request**.
  - Computes two-mode surface water area quantification (`calculate_water_area_km2`).
  - Returns probability distribution diagnostics (`min`, `max`, `mean`, `std`, `is_saturated`).

### C. Docker Microservice (`docker/Dockerfile`, `docker-compose.yml`)
- Base image: `python:3.11-slim`.
- Dedicated non-root user `appuser` (UID 10001).
- Healthcheck configured on `/health` via `curl` with 3 retries.
- Environment variable `PYTHONPATH=/app` and `PORT=8000`.

---

## 3. Hugging Face Hub LFS Repository Audit

Remote repository: `https://huggingface.co/safalya06/geovision-water-segmentation-models`  
Commit: `441a99034b36336c34d86c23cfe9c1a8114ca3a4`

| Checkpoint Name | Architecture | Remote Size (Bytes) | Authoritative SHA-256 Hash | Integrity |
|---|---|---|---|---|
| `fpn_best.pth` | FPN-ResNet34 (Champion) | 92,779,733 | `d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1` | **MATCH (100%)** |
| `unet_plusplus_best.pth` | UNet++-ResNet34 | 104,528,035 | `497e246ef79d87f3b2255cc119e7ab8bd0045d48caf3d4710e9f6738e56c4548` | **MATCH (100%)** |
| `deeplabv3_plus_best.pth` | DeepLabV3+-ResNet34 | 89,936,639 | `6c7c6dd4835d29be34b9eda03bdfca878914749a526c44cade99906cc1b1b827` | **MATCH (100%)** |
| `segformer_best.pth` | SegFormer-MiT-B0 | 14,949,211 | `2c762a52a5aa5cafb6aecb698bf483257fd5279bce3f3af7368850ba9354e986` | **MATCH (100%)** |

All four checkpoints are stored in native Git LFS format and match local experiment checkpoints bit-for-bit.
