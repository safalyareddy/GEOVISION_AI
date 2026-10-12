# FastAPI Microservice API Reference

The GeoVision AI REST API provides high-throughput inference endpoints for satellite water segmentation.

---

## 1. Service Launch

```bash
# Launch FastAPI via Uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```
* **Interactive OpenAPI Swagger Docs**: `http://localhost:8000/docs`
* **Interactive Redoc**: `http://localhost:8000/redoc`

---

## 2. Endpoints

### `GET /health`
Returns microservice health and loaded neural network models.
* **Status**: `200 OK`
* **Response**:
```json
{
  "status": "ok",
  "torch_available": true,
  "loaded_models": ["unet_plusplus"]
}
```

### `GET /models`
Lists supported architectures.
* **Status**: `200 OK`
* **Response**:
```json
{
  "supported_models": {
    "unet_plusplus": "UNet++ (Nested UNet for multi-scale feature fusion)",
    "deeplabv3_plus": "DeepLabV3+ (Atrous Spatial Pyramid Pooling)",
    "segformer": "SegFormer-B2 (Lightweight Vision Transformer for Segmentation)"
  },
  "default_model": "unet_plusplus"
}
```

### `POST /predict`
Uploads a satellite image tile and returns the segmentation mask and metrics.
* **Content-Type**: `multipart/form-data`
* **Parameters**:
  * `file`: Binary image file (`PNG`, `JPEG`, `TIFF`, max 10 MB)
  * `model_name`: `unet_plusplus` | `deeplabv3_plus` | `segformer`
  * `threshold`: Float between `0.0` and `1.0` (default: `0.5`)
* **Response (`200 OK`)**:
```json
{
  "model_name": "unet_plusplus",
  "threshold": 0.5,
  "water_coverage_percentage": 24.15,
  "water_pixels": 15827,
  "total_pixels": 65536,
  "latency_ms": 19.33,
  "image_dimensions": [256, 256],
  "mask_base64": "iVBORw0KGgoAAAANSUhEUgAAAQ..."
}
```
* **Error Responses**:
  * `400 Bad Request`: Empty file, invalid image bytes, unsupported MIME type, or threshold out of range.
  * `413 Payload Too Large`: Upload exceeds 10 MB limit.
  * `503 Service Unavailable`: Model architecture failed to initialize.
