# Final Inference & API Verification Audit

**Audit Date**: October 13, 2026  
**Auditor**: Backend & Machine Learning API Engineer  
**Scope**: Microservice Endpoints (`/`, `/health`, `/models`, `/predict`), Error Handling, Non-Standard Dimensions, Concurrency

---

## 1. REST Microservice Endpoints Verification

Inference was tested against the official FastAPI microservice (`src/api/main.py`) using `starlette.testclient.TestClient`:

| Endpoint | HTTP Method | Expected Status | Actual Status | Response Payload Summary | Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| `/` | `GET` | 200 | **200** | Service name, version 1.0.0, supported model architectures. | **PASS — VERIFIED** |
| `/health` | `GET` | 200 | **200** | `{"status": "ok", "torch_available": true, "loaded_models": [...]}` | **PASS — VERIFIED** |
| `/models` | `GET` | 200 | **200** | Supported architectures dictionary, default model `unet_plusplus`. | **PASS — VERIFIED** |
| `/predict` (Valid Satellite Tile) | `POST` | 200 | **200** | Latency 418.3 ms, water coverage 4.09%, base64 PNG mask string. | **PASS — VERIFIED** |

---

## 2. Mask Reconstruction & Dimensions Fidelity

* **Image Input Dimensions**: $256 \times 256 \times 3$ (RGB)
* **Returned JSON Dimensions Field**: `[256, 256]`
* **Base64 Decoded Mask**:
  - Image Format: PNG
  - Color Mode: `L` (8-bit grayscale binary mask, $\{0, 255\}$)
  - Dimensions: $256 \times 256$ pixels (exact 1:1 spatial match to input)
* **Non-Standard Input Resolution ($512 \times 512$)**:
  - Successfully downsampled to $256 \times 256$ for model forward pass.
  - Predicted mask nearest-neighbor upsampled back to original dimensions ($512 \times 512$).
  - Returned dimensions: `[512, 512]`.
* **Grayscale Input (1-channel)**:
  - Automatically converted via `PIL.Image.convert("RGB")` without crashes.
  - Returned status code 200 with valid segmentation mask.

---

## 3. Security, Validation & Error Handling

| Negative / Edge Test Case | Input Characteristics | Expected HTTP Code | Actual HTTP Code | Error Response Message | Status |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Invalid Model Architecture** | `model_name="non_existent_net"` | 400 | **400** | "Invalid model 'non_existent_net'. Choose from: ..." | **PASS — VERIFIED** |
| **Invalid Threshold (High)** | `threshold=1.2` | 400 | **400** | "Threshold must be a float between 0.0 and 1.0." | **PASS — VERIFIED** |
| **Invalid Threshold (Low)** | `threshold=-0.1` | 400 | **400** | "Threshold must be a float between 0.0 and 1.0." | **PASS — VERIFIED** |
| **Empty File Upload** | 0 bytes (`b""`) | 400 | **400** | "Uploaded image file is empty (0 bytes)." | **PASS — VERIFIED** |
| **Corrupt Image Payload** | Malformed ASCII byte stream | 400 | **400** | "Invalid or corrupt image payload. Could not decode image." | **PASS — VERIFIED** |
| **Unsupported MIME Type** | `text/plain` | 400 | **400** | "Unsupported image file format 'text/plain'." | **PASS — VERIFIED** |
| **Oversized Payload (>10 MB)**| $10\text{MB} + 1\text{KB}$ stream | 413 | **413** | "Uploaded file exceeds maximum size limit of 10 MB." | **PASS — VERIFIED** |

---

## 4. Concurrency & Resource Cleanliness

* **Test**: 10 simultaneous inference requests dispatched across 5 worker threads (`ThreadPoolExecutor`).
* **Result**: **10 / 10 succeeded with HTTP 200**.
* **Memory & State**: Inference strictly wrapped in `torch.no_grad()`, cached model shared in read-only evaluation mode without thread-safety memory leaks.
