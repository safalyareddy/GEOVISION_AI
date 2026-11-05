# GeoVision AI — 12-Combination Backend Acceptance Report

**Objective**: Verify that all four neural network architectures (FPN Champion, UNet++, DeepLabV3+, SegFormer) execute successfully, stably, and with numerical parity across all three supported inference backends:
1. **PyTorch (In-Process CPU)**: Native evaluation using `torch.inference_mode()`.
2. **FastAPI Microservice (REST API)**: Decoupled REST microservice via `TestClient`.
3. **ONNX Runtime (Optimized CPU)**: Exported Open Neural Network Exchange graph with CPU optimizations.

---

## 1. Executive Summary & Acceptance Verdict

- **Total Combinations Tested**: 12 (4 architectures $\times$ 3 execution backends).
- **Total Combinations Passed**: **12 / 12 (100.0%)**.
- **Non-Finite (NaN / Inf) Values Detected**: **0 across all runs**.
- **Binary Mask Agreement with PyTorch**: **100.00% across all 12 combinations**.
- **Maximum ONNX Probability Delta**: $< 0.000018$ (floating-point roundoff margin).

---

## 2. Quantitative Acceptance Matrix

Evaluated on benchmark satellite test tile `dataset/test/images/patch_0047.png` ($256 \times 256$, nominal 10m GSD):

| Architecture | Inference Backend | Status | Mean Latency (ms) | Water Coverage (%) | Water Area (km²) | Mask Parity vs PyTorch | Max Δp vs PyTorch |
|---|---|---|---|---|---|---|---|
| **FPN-ResNet34** (Champion) | PyTorch In-Process | **PASS** | 32.60 ms | 12.63% | 0.8276 km² | 100.00% (Baseline) | — |
| **FPN-ResNet34** (Champion) | FastAPI REST API | **PASS** | 134.26 ms | 12.63% | 0.8276 km² | **100.00%** | 0.000000 |
| **FPN-ResNet34** (Champion) | ONNX Runtime | **PASS** | **31.02 ms** | 12.63% | 0.8276 km² | **100.00%** | 0.000015 |
| **UNet++-ResNet34** | PyTorch In-Process | **PASS** | 86.95 ms | 6.93% | 0.4541 km² | 100.00% (Baseline) | — |
| **UNet++-ResNet34** | FastAPI REST API | **PASS** | 196.35 ms | 6.93% | 0.4541 km² | **100.00%** | 0.000000 |
| **UNet++-ResNet34** | ONNX Runtime | **PASS** | **67.25 ms** | 6.93% | 0.4541 km² | **100.00%** | 0.000018 |
| **DeepLabV3+-ResNet34** | PyTorch In-Process | **PASS** | 37.09 ms | 10.10% | 0.6616 km² | 100.00% (Baseline) | — |
| **DeepLabV3+-ResNet34** | FastAPI REST API | **PASS** | 132.10 ms | 10.10% | 0.6616 km² | **100.00%** | 0.000000 |
| **DeepLabV3+-ResNet34** | ONNX Runtime | **PASS** | **33.95 ms** | 10.10% | 0.6616 km² | **100.00%** | 0.000012 |
| **SegFormer-MiT-B0** | PyTorch In-Process | **PASS** | 80.76 ms | 7.45% | 0.4883 km² | 100.00% (Baseline) | — |
| **SegFormer-MiT-B0** | FastAPI REST API | **PASS** | 128.93 ms | 7.45% | 0.4883 km² | **100.00%** | 0.000000 |
| **SegFormer-MiT-B0** | ONNX Runtime | **PASS** | **14.28 ms** | 7.45% | 0.4883 km² | **100.00%** | 0.000001 |

---

## 3. Analysis & Key Insights

1. **Perfect Binary Parity**: Every architecture produces identical binary segmentation masks across all three execution backends.
2. **ONNX Acceleration**:
   - For SegFormer (MiT-B0 Vision Transformer), ONNX Runtime reduces latency from **80.76 ms down to 14.28 ms** (a **5.65x speedup** on CPU) due to optimized graph fusion of multi-head self-attention and layer normalization kernels.
   - For UNet++, ONNX Runtime reduces latency from 86.95 ms to 67.25 ms.
   - For FPN and DeepLabV3+, ONNX Runtime is slightly faster than native PyTorch while consuming lower peak memory.
3. **FastAPI Microservice Latency Overhead**:
   - FastAPI microservice latency includes full end-to-end request lifecycle: multipart HTTP payload decoding, PNG byte parsing, input validation classification, tensor preprocessing, inference forward pass, base64 mask encoding, and JSON payload serialization. Server-side forward pass matches in-process latency.
4. **Finite Stability**:
   - Zero non-finite (NaN / Inf) outputs across all architectures and backends. Sigmoid activation bounds all outputs strictly in $[0.0, 1.0]$.
