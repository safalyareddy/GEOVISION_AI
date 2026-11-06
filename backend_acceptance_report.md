# GeoVision AI — 8-Combination Backend Acceptance Report

**Objective**: Verify that all four neural network architectures (FPN Champion, UNet++, DeepLabV3+, SegFormer) execute successfully, stably, and with numerical parity across both production inference engines:
1. **PyTorch (In-Process CPU / Apple Silicon MPS)**: Native evaluation using `torch.inference_mode()` with strict cryptographic state-dict verification.
2. **ONNX Runtime (Accelerated CPU)**: Standalone Open Neural Network Exchange computational graphs with CPU kernel optimizations.

*(Architectural Notice: The legacy FastAPI microservice has been completely retired from the active runtime architecture to eliminate unnecessary HTTP serialization overhead, proxy hops, and multi-process server dependencies).*

---

## 1. Executive Summary & Acceptance Verdict

- **Total Combinations Tested**: 8 (4 architectures $\times$ 2 execution engines).
- **Total Combinations Passed**: **8 / 8 (100.0%)**.
- **Non-Finite (NaN / Inf) Values Detected**: **0 across all runs**.
- **Binary Mask Agreement with PyTorch**: **100.00% across all 8 combinations and all 7 test scenes**.
- **Maximum ONNX Probability Delta**: $< 3.18 \times 10^{-5}$ (floating-point precision margin).
- **Mean Absolute Probability Delta**: $\approx 1.1 \times 10^{-7}$.

---

## 2. Multi-Tile Parity Acceptance Matrix

Evaluated using `scripts/verify_8_combinations.py` across 7 diverse Sentinel-2 test scenes (low-water `<1%`, mixed land/water `10–15%`, complex boundary demo patch `25%`, and water-rich `>45%`):

| Architecture | Engine | Status | Binary Mask Parity (7 Tiles) | Max Abs Δp | Mean Abs Δp | Mean MSE |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **FPN-ResNet34** (Champion) | PyTorch In-Process | **PASS** | 100.00% (Baseline) | — | — | — |
| **FPN-ResNet34** (Champion) | ONNX Runtime | **PASS** | **100.00%** | $1.69 \times 10^{-5}$ | $1.21 \times 10^{-7}$ | $5.01 \times 10^{-13}$ |
| **UNet++-ResNet34** | PyTorch In-Process | **PASS** | 100.00% (Baseline) | — | — | — |
| **UNet++-ResNet34** | ONNX Runtime | **PASS** | **100.00%** | $3.18 \times 10^{-5}$ | $1.11 \times 10^{-7}$ | $8.18 \times 10^{-13}$ |
| **DeepLabV3+-ResNet34** | PyTorch In-Process | **PASS** | 100.00% (Baseline) | — | — | — |
| **DeepLabV3+-ResNet34** | ONNX Runtime | **PASS** | **100.00%** | $1.39 \times 10^{-5}$ | $9.09 \times 10^{-8}$ | $2.26 \times 10^{-13}$ |
| **SegFormer-MiT-B0** | PyTorch In-Process | **PASS** | 100.00% (Baseline) | — | — | — |
| **SegFormer-MiT-B0** | ONNX Runtime | **PASS** | **100.00%** | $1.73 \times 10^{-6}$ | $1.30 \times 10^{-7}$ | $3.64 \times 10^{-14}$ |

---

## 3. Benchmark Tile Performance (Patch 0047 — Difficult Boundary Scene)

Evaluated on benchmark satellite test tile `dataset/test/images/patch_0047.png` ($256 \times 256$, nominal 10m GSD):

| Architecture | Inference Engine | Status | Mean Latency (ms) | Water Coverage (%) | Water Area (km²) | Mask Parity vs PyTorch | Max Δp vs PyTorch |
|---|---|---|---|---|---|---|---|
| **FPN-ResNet34** (Champion) | PyTorch In-Process | **PASS** | 28.32 ms | 12.63% | 0.8276 km² | 100.00% (Baseline) | — |
| **FPN-ResNet34** (Champion) | ONNX Runtime | **PASS** | **33.25 ms** | 12.63% | 0.8276 km² | **100.00%** | 0.000015 |
| **UNet++-ResNet34** | PyTorch In-Process | **PASS** | 207.40 ms | 6.93% | 0.4541 km² | 100.00% (Baseline) | — |
| **UNet++-ResNet34** | ONNX Runtime | **PASS** | **63.87 ms** | 6.93% | 0.4541 km² | **100.00%** | 0.000018 |
| **DeepLabV3+-ResNet34** | PyTorch In-Process | **PASS** | 67.14 ms | 10.10% | 0.6616 km² | 100.00% (Baseline) | — |
| **DeepLabV3+-ResNet34** | ONNX Runtime | **PASS** | **34.24 ms** | 10.10% | 0.6616 km² | **100.00%** | 0.000012 |
| **SegFormer-MiT-B0** | PyTorch In-Process | **PASS** | 63.11 ms | 7.45% | 0.4883 km² | 100.00% (Baseline) | — |
| **SegFormer-MiT-B0** | ONNX Runtime | **PASS** | **12.47 ms** | 7.45% | 0.4883 km² | **100.00%** | 0.000001 |

---

## 4. Key Engineering Insights

1. **Perfect Binary Parity**: Every architecture produces identical binary segmentation masks across both execution engines (100.00% mask agreement).
2. **Vision Transformer Acceleration**:
   - For **SegFormer (MiT-B0 Vision Transformer)**, ONNX Runtime reduces latency from **63.11 ms down to 12.47 ms** (a **$5.06\times$ speedup** on CPU) due to optimized graph fusion of multi-head self-attention and layer normalization operations.
   - For **UNet++**, ONNX Runtime reduces latency from 207.40 ms to 63.87 ms (a **$3.25\times$ speedup**).
   - For **DeepLabV3+**, ONNX Runtime reduces latency from 67.14 ms to 34.24 ms (a **$1.96\times$ speedup**).
3. **Finite Stability**:
   - Zero non-finite (NaN / Inf) outputs across all architectures and engines. Sigmoid activation strictly bounds all outputs in $[0.0, 1.0]$.
4. **FastAPI Retirement Benefit**:
   - Eliminating the FastAPI microservice removed 100+ ms of HTTP serialization/deserialization latency, eliminated Docker multi-process dependencies, and simplified production deployment down to a single clean Streamlit container.
