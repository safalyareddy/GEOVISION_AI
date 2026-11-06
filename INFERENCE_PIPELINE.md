# GeoVision AI — Unified Multi-Engine Inference Pipeline Specification (PyTorch & ONNX Runtime)

This document details the unified production inference architecture implemented in [`src/inference.py`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/src/inference.py).

---

## 1. Unified Interface Architecture

All segmentation models adhere to a standard, functional Python interface guaranteeing consistent preprocessing, device allocation, cryptographic verification, and deterministic execution across both native **PyTorch** and accelerated **ONNX Runtime** inference engines.

```
+-----------------------------------------------------------------------------------------+
|                                    Client Application                                   |
|                        (Streamlit Interactive UI Dashboard :7860)                       |
|               *(FastAPI microservice has been fully retired from runtime)*              |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                               src.inference Unified API                                 |
|                                                                                         |
|  • load_model(model_name, device)                                                       |
|  • load_onnx_session(model_name)                                                        |
|  • predict(image, model_name, threshold, device, engine="pytorch"|"onnx")              |
|  • predict_onnx(image, model_name, threshold)                                           |
|  • predict_all(image, threshold, device, engine="pytorch"|"onnx")                       |
|  • get_model_metadata(model_name)                                                       |
|  • get_model_status(model_name)                                                         |
+-----------------------------------------------------------------------------------------+
           |                                                           |
           v                                                           v
+-----------------------+                                   +-----------------------+
|  src.models Registry  |                                   | Preprocessing & Tensors|
|  • FPN (ResNet34)     |                                   | • PIL RGB Conversion  |
|  • UNet++ (ResNet34)  |                                   | • Resize (256, 256)   |
|  • DeepLabV3+ (RN34)  |                                   | • Float32 [0.0, 1.0]  |
|  • SegFormer (MiT-B0) |                                   | • NCHW Tensor format  |
+-----------------------+                                   +-----------------------+
           |                                                           |
           +-----------------------------+-----------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
   +-----------------------------------+   +-----------------------------------+
   |      PyTorch Forward Engine       |   |    ONNX Runtime Accelerated Engine|
   |  • torch.inference_mode()         |   |  • In-Process CPU Execution       |
   |  • model.eval() guarantees        |   |  • Fused Attention & LayerNorm    |
   |  • CPU / Apple Silicon MPS Target |   |  • Opset 14 / 17 Graph            |
   +-----------------------------------+   +-----------------------------------+
                     |                                       |
                     +-------------------+-------------------+
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
- Validates model architecture against supported models (`fpn`, `unet_plusplus`, `deeplabv3_plus`, `segformer`).
- Verifies checkpoint file existence and SHA-256 cryptographic checksum.
- Strictly loads state dictionary (`strict=True`).
- Freezes running statistics with `model.eval()`.
- Caches model in memory (`_MODEL_CACHE[(model_name, device)]`) to prevent redundant disk loads.

### `load_onnx_session(model_name: str = "fpn") -> ort.InferenceSession`
- Validates model architecture against supported models.
- Resolves exported ONNX graph path (`checkpoints/onnx/<model_name>.onnx` or `checkpoints/<model_name>_best.onnx`).
- If missing, automatically generates the ONNX computational graph using PyTorch export with dual fallback (`dynamo=False`, opset 14/17).
- Initializes and caches `onnxruntime.InferenceSession` with `CPUExecutionProvider`.

### `predict(image, model_name="fpn", threshold=0.50, device=None, engine="pytorch") -> Dict[str, Any]`
Accepts PIL Image, NumPy array, or file path. Routes execution to the selected engine (`"pytorch"` or `"onnx"`). Returns a dictionary containing:
- `model_name`: Architecture used.
- `engine`: Inference engine (`"pytorch"` or `"onnx"`).
- `device`: Hardware target (`cpu`, `mps`).
- `threshold`: Binary decision cutoff ($\tau$).
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

### `predict_onnx(image, model_name="fpn", threshold=0.50) -> Dict[str, Any]`
Executes inference directly through the optimized ONNX Runtime graph session, returning the same standardized schema with high-throughput CPU acceleration.

### `predict_all(image, threshold=0.50, device=None, engine="pytorch") -> Dict[str, Dict[str, Any]]`
Executes multi-model inference sequentially across all four verified models on the same image using the specified engine, allowing instantaneous cross-architecture comparison.

### `get_model_metadata(model_name: str) -> Dict[str, Any]`
Returns verified metadata: display name, architecture family, encoder, parameter count, input size, normalization policy, and SHA-256 hash.

### `get_model_status(model_name: Optional[str] = None) -> Dict[str, Any]`
Returns current availability, local presence, Hugging Face Hub status, and whether the model instance is currently cached in memory.

---

## 3. Preprocessing Specifications

All four models were trained on RGB satellite tiles converted to float32 tensors scaled to $[0.0, 1.0]$. The inference pipeline enforces identical preprocessing parity across both engines:

```python
import torchvision.transforms as T
from PIL import Image

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()  # Automatically converts PIL RGB image to [0.0, 1.0] float32 tensor
])
```

---

## 4. Hardware Acceleration & Parity

1. **Native PyTorch Engine**:
   - Supports **CPU** and Apple Silicon **MPS** (Metal Performance Shaders).
   - Numerical equivalence between CPU and MPS passes has been validated:
     $$\text{Mask}_{\text{MPS}} \equiv \text{Mask}_{\text{CPU}}$$
2. **ONNX Runtime Engine**:
   - Standalone CPU graph execution with fused attention kernels.
   - 100.0% binary mask agreement across all 4 architectures and 7 diverse Sentinel-2 test scenes.
   - Max absolute probability difference: $\max |\Delta p| < 3.2 \times 10^{-5}$ (floating-point precision margin).

---

## 5. Usage Examples

### Example 1: PyTorch In-Process Inference
```python
from src.inference import predict

result = predict(
    image="dataset/test/images/patch_0047.png",
    model_name="fpn",
    threshold=0.50,
    engine="pytorch"
)

print(f"Engine: {result['engine']} | Model: {result['model_name']}")
print(f"Latency: {result['latency_ms']:.1f} ms")
print(f"Water Coverage: {result['water_coverage_pct']:.2f}%")
print(f"Estimated Water Area: {result['water_area_km2']:.4f} km²")
```

### Example 2: ONNX Runtime Accelerated Inference
```python
from src.inference import predict

result_onnx = predict(
    image="dataset/test/images/patch_0047.png",
    model_name="segformer",
    threshold=0.50,
    engine="onnx"
)

print(f"Engine: {result_onnx['engine']} | Latency: {result_onnx['latency_ms']:.1f} ms")
print(f"Water Coverage: {result_onnx['water_coverage_pct']:.2f}%")
```

### Example 3: Side-by-Side Comparison Across All 4 Models
```python
from src.inference import predict_all

results = predict_all("dataset/test/images/patch_0047.png", threshold=0.50, engine="onnx")

for name, res in results.items():
    print(f"{name:<15} -> Coverage: {res['water_coverage_pct']:5.2f}% | Latency: {res['latency_ms']:5.1f} ms")
```

---

## 6. ONNX Export & Fallback Mechanism

When exporting PyTorch modules to ONNX graphs, `src/inference.py` and `scripts/export_onnx.py` employ a robust fallback strategy:
1. **Primary Attempt**: Standard Torch ONNX export with `opset_version=17`.
2. **Fallback Attempt**: If `onnxscript` / Dynamo export encounters unsupported operators or missing optional packages, the pipeline cleanly falls back to legacy Tracer export (`dynamo=False`) with `opset_version=14`.
3. **Graph Storage**: Large models with weight tensors exceeding standard protobuf limits automatically store external weight sidecars (`.onnx.data`) in `checkpoints/onnx/`.
