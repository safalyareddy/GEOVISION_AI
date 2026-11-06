# GeoVision AI — API Reference & Programming Interface

This document specifies the primary Python programmatic API (`src.inference`, `src.utils.input_validator`) used for production serving, along with historical reference for the retired REST endpoints.

> [!IMPORTANT]
> **Architecture Notice**: The FastAPI REST microservice has been completely retired from the active production runtime. All inference, cryptographic checkpoint verification, and input validation now execute in-process via `src.inference`.

---

## 1. Primary Programmatic Python API (`src.inference`)

### `predict(image, model_name="fpn", threshold=0.50, device=None, engine="pytorch") -> Dict[str, Any]`
Executes single-model inference on a satellite image tile using either the native PyTorch engine or the accelerated ONNX Runtime engine.

**Parameters**:
- `image`: Input payload — accepts a PIL `Image.Image`, NumPy array (`np.ndarray`), or file path (`str` / `Path`).
- `model_name`: Architecture identifier (`"fpn"`, `"unet_plusplus"`, `"deeplabv3_plus"`, `"segformer"`). Default: `"fpn"`.
- `threshold`: Binary decision threshold $\tau \in [0.0, 1.0]$. Default: `0.50`.
- `device`: Hardware target (`"cpu"`, `"mps"`). Defaults to optimal detected device.
- `engine`: Inference engine backend — `"pytorch"` (default) or `"onnx"`.

**Returns**:
A structured dictionary containing:
```python
{
    "model_name": "fpn",
    "engine": "pytorch",              # or "onnx"
    "device": "cpu",
    "threshold": 0.50,
    "latency_ms": 28.32,
    "water_coverage_pct": 12.63,
    "water_area_km2": 0.8276,
    "water_pixels": 8276,
    "total_pixels": 65536,
    "original_dimensions": [256, 256],
    "mask_array_256": np.ndarray,     # (256, 256) binary {0, 1}
    "probability_map_256": np.ndarray,# (256, 256) float32 in [0.0, 1.0]
    "mask_image": PIL.Image.Image,    # Grayscale mask at original dimensions
    "binary_mask": np.ndarray         # Binary mask at original dimensions
}
```

---

### `predict_onnx(image, model_name="fpn", threshold=0.50) -> Dict[str, Any]`
Convenience wrapper that executes inference directly through an accelerated ONNX Runtime CPU graph session, returning the identical output schema.

---

### `predict_all(image, threshold=0.50, device=None, engine="pytorch") -> Dict[str, Dict[str, Any]]`
Executes multi-model inference sequentially across all four verified models (`fpn`, `unet_plusplus`, `deeplabv3_plus`, `segformer`) on the same image, allowing instantaneous cross-architecture comparison.

---

### `load_model(model_name="fpn", device="cpu") -> torch.nn.Module`
Loads a verified PyTorch model instance:
- Verifies local presence or downloads from Hugging Face Hub (`safalya06/geovision-water-segmentation-models@441a990`).
- Validates cryptographic SHA-256 hash.
- Deserializes with `weights_only=True` and applies `model.load_state_dict(..., strict=True)`.
- Freezes running statistics with `model.eval()`.
- Caches model in memory (`_MODEL_CACHE`).

---

### `load_onnx_session(model_name="fpn") -> onnxruntime.InferenceSession`
Loads or generates the optimized ONNX computation graph:
- Resolves `checkpoints/onnx/<model_name>.onnx`.
- If missing, automatically generates the graph from PyTorch weights with opset 14/17 fallback.
- Initializes and caches `onnxruntime.InferenceSession` with `CPUExecutionProvider`.

---

## 2. Scientific Input Validation API (`src.utils.input_validator`)

### `validate_input_image(image_bytes: bytes, filename: str) -> ValidationResult`
Inspects an inbound file payload prior to tensor processing to prevent unscientific hydrological claims.

**Classification States**:
- `VERIFIED_PROJECT_SAMPLE`: Payload matches curated Sentinel-2 test split patch by SHA-256 hash.
- `VERIFIED_GEOSPATIAL_INPUT`: GeoTIFF with embedded coordinate reference system and pixel scale tags.
- `USER_DECLARED_SATELLITE_UNVERIFIED`: Plausible satellite imagery lacking embedded spatial tags.
- `UNSUPPORTED_INPUT` / `UNSUPPORTED_PHOTOGRAPH`: Consumer photograph (iPhone, Canon, Nikon EXIF tags) or solid black/white image.
- `INVALID_IMAGE`: Corrupted bytes, zero-length file, or unsupported MIME type.

---

## 3. Retired REST API Specification (Legacy Reference)

*(Archival Note: Existed prior to commit `9eddec5` when FastAPI was retired).*

- `GET /health`: Returned service status and loaded PyTorch models.
- `GET /models`: Listed supported model architectures.
- `POST /predict`: Accepted multipart/form-data image uploads with model name and threshold query parameters.
