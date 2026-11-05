# GeoVision AI — Common PyTorch Inference Pipeline Specification

This document details the unified production inference architecture implemented in [`src/inference.py`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/src/inference.py).

---

## 1. Unified Interface Architecture

All segmentation models adhere to a standard, functional Python interface guaranteeing consistent preprocessing, device allocation, cryptographic verification, and deterministic execution.

```
+-----------------------------------------------------------------------------------------+
|                                    Client Application                                   |
|                          (FastAPI REST Microservice / Streamlit UI)                     |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                               src.inference Unified API                                 |
|                                                                                         |
|  • load_model(model_name, device)                                                       |
|  • predict(image, model_name, threshold, device)                                        |
|  • predict_all(image, threshold, device)                                               |
|  • get_model_metadata(model_name)                                                       |
|  • get_model_status(model_name)                                                         |
+-----------------------------------------------------------------------------------------+
            |                                                      |
            v                                                      v
+-----------------------+                              +-----------------------+
|  src.models Registry  |                              | Preprocessing & Tensors|
|  • UNet++ (ResNet34)  |                              | • PIL RGB Conversion  |
|  • DeepLabV3+ (RN34)  |                              | • Resize (256, 256)   |
|  • SegFormer (MiT-B0) |                              | • Float32 [0.0, 1.0]  |
|  • FPN (ResNet34)     |                              | • NCHW Tensor format  |
+-----------------------+                              +-----------------------+
            |                                                      |
            +--------------------------+---------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |      PyTorch Forward Engine       |
                     |  • torch.inference_mode()         |
                     |  • model.eval() guarantees        |
                     |  • CPU / Apple Silicon MPS Target |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |    Postprocessing & Analytics     |
                     |  • Sigmoid Activation             |
                     |  • Binary Decision Cutoff (τ)     |
                     |  • Nominal Sentinel-2 10m GSD     |
                     |  • Water Coverage % & Area (km²)  |
                     |  • Original Dimensions Rescaling  |
                     +-----------------------------------+
```

---

## 2. API Function Signatures & Behavior

### `load_model(model_name: str = "fpn", device: str = "cpu") -> torch.nn.Module`
- Validates model architecture against supported models.
- Verifies checkpoint file existence and SHA-256 cryptographic checksum.
- Strictly loads state dictionary (`strict=True`).
- Freezes running statistics with `model.eval()`.
- Caches model in memory (`_MODEL_CACHE[(model_name, device)]`) to prevent redundant disk loads.

### `predict(image, model_name="fpn", threshold=0.50, device=None) -> Dict[str, Any]`
Accepts PIL Image, NumPy array, or file path. Returns a dictionary containing:
- `model_name`: Architecture used.
- `device`: Hardware target (`cpu`, `mps`).
- `threshold`: Binary decision cutoff.
- `latency_ms`: Execution time in milliseconds.
- `water_coverage_pct`: Fraction of tile classified as water ($0.0\% - 100.0\%$).
- `water_area_km2`: Hydrological area under nominal 10m GSD ($100\text{ m}^2/\text{pixel}$).
- `water_pixels`: Integer count of positive water pixels.
- `total_pixels`: Total spatial pixels ($65,536$ for $256 \times 256$).
- `original_dimensions`: `[width, height]` of uploaded image.
- `mask_array_256`: NumPy binary array $(256 \times 256)$ with values in $\{0, 1\}$.
- `probability_map_256`: Continuous float32 sigmoid activation map $\in [0.0, 1.0]$.
- `mask_image`: PIL grayscale image rescaled back to original image dimensions.
- `binary_mask`: NumPy binary array matching original input dimensions.

### `predict_all(image, threshold=0.50, device=None) -> Dict[str, Dict[str, Any]]`
Executes multi-model inference sequentially across all four verified models on the same image, allowing instantaneous cross-architecture comparison.

### `get_model_metadata(model_name: str) -> Dict[str, Any]`
Returns verified metadata: display name, architecture family, encoder, parameter count, input size, normalization policy, and SHA-256 hash.

### `get_model_status(model_name: Optional[str] = None) -> Dict[str, Any]`
Returns current availability, local presence, Hugging Face Hub status, and whether the model instance is currently cached in memory.

---

## 3. Preprocessing Specifications

All four models were trained on RGB satellite tiles converted to float32 tensors scaled to $[0.0, 1.0]$. The inference pipeline enforces identical preprocessing parity:

```python
import torchvision.transforms as T
from PIL import Image

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()  # Automatically converts PIL RGB image to [0.0, 1.0] float32 tensor
])
```

---

## 4. Hardware Acceleration (CPU & Apple Silicon MPS)

Device selection dynamically detects Apple Silicon Metal Performance Shaders (MPS):

```python
from src.inference import get_optimal_device

# Automatically selects 'mps' on Apple Silicon Mac when requested, with graceful CPU fallback
device = get_optimal_device(prefer_mps=True)
```

Numerical equivalence between CPU and MPS passes has been validated in the test suite:
$$\text{Mask}_{\text{MPS}} \equiv \text{Mask}_{\text{CPU}}$$

---

## 5. Usage Examples

### Example 1: Local Script Inference
```python
from pathlib import Path
from src.inference import predict

result = predict(
    image="dataset/test/images/patch_0000.png",
    model_name="fpn",
    threshold=0.50,
    device="cpu"
)

print(f"Model: {result['model_name']}")
print(f"Latency: {result['latency_ms']:.1f} ms")
print(f"Water Coverage: {result['water_coverage_pct']:.2f}%")
print(f"Estimated Water Area: {result['water_area_km2']:.4f} km²")
```

### Example 2: Side-by-Side Comparison Across All 4 Models
```python
from src.inference import predict_all

results = predict_all("dataset/test/images/patch_0000.png", threshold=0.50)

for name, res in results.items():
    print(f"{name:<15} -> Coverage: {res['water_coverage_pct']:5.2f}% | Latency: {res['latency_ms']:5.1f} ms")
```
