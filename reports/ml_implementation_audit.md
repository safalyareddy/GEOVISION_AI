# 🧠 GeoVision AI — Machine Learning Implementation Audit Report (Phase 5)

**Date**: 2026-10-12  
**Auditor**: Senior Machine Learning & MLOps Engineer  
**Pipeline**: Raw imagery → preprocessing → dataset loader → model → loss → optimizer → training → validation → test evaluation → ONNX export → FastAPI/Streamlit inference  

---

## 1. Architectural & Tensor Specification Analysis

| Component | Specification | Code Verification Path | Status |
| :--- | :--- | :--- | :---: |
| **Input Shape** | $[B, 3, 256, 256]$ (RGB, FloatTensor $[0.0, 1.0]$) | `src/data/dataset.py:228-235` | **VERIFIED** |
| **Target Mask Shape**| $[B, 1, 256, 256]$ (Binary FloatTensor $\{0.0, 1.0\}$)| `src/data/dataset.py:237-240` | **VERIFIED** |
| **Output Shape** | $[B, 1, 256, 256]$ (Raw unconstrained logits) | `src/models.py:65, 93, 124` | **VERIFIED** |
| **Output Activation** | Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}} \in [0.0, 1.0]$ | `src/evaluate.py:133`, `src/api/main.py:151` | **VERIFIED** |
| **Decision Threshold**| Configurable $\tau \in [0.10, 0.90]$ (Default $\tau = 0.50$)| `app.py:126`, `src/evaluate.py:30` | **VERIFIED** |
| **Loss Function** | $\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{Dice}} + 0.5 \cdot \mathcal{L}_{\text{Focal}}$ | `src/loss.py:80-105` | **VERIFIED** |
| **Optimizer** | AdamW ($\text{lr} = 10^{-3}, \text{weight\_decay} = 10^{-4}$) | `src/train.py:110-118` | **VERIFIED** |
| **LR Scheduler** | CosineAnnealingLR ($T_{\max} = 10$) | `src/train.py:111-118` | **VERIFIED** |
| **Hardware Device** | Apple Silicon M4 Metal Performance Shaders (`mps`) | Verified ($23.09\text{ s}$ per epoch) | **VERIFIED** |

---

## 2. Loss Function Formulation & Imbalance Correctness

Satellite water segmentation is characterized by severe class imbalance (5.34% water vs 94.66% land). GeoVision AI implements a custom joint loss:

$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{Dice}} + 0.5 \cdot \mathcal{L}_{\text{Focal}}$$

1. **Dice Loss**:
   $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i g_i + \epsilon}{\sum p_i^2 + \sum g_i^2 + \epsilon}$$
   Directly optimizes regional overlap and contour alignment, invariant to background pixel volume.
2. **Focal Loss**:
   $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   Down-weights easy well-classified land pixels ($p_t \approx 1$) with modulating factor $\gamma = 2.0$ and focuses gradient updates on difficult shoreline and river boundaries.
3. **Numerics & Stability**:
   - Both loss terms operate strictly on raw logits (`from_logits=True`) using log-sum-exp stabilization to avoid vanishing gradients or numerical overflow.

---

## 3. Root Cause Analysis: Fallback Model vs. Checkpoint Weight Parity

A pivotal diagnostic discovery was made regarding model evaluation:
- Previously, `segmentation-models-pytorch` (`smp`) was not present in the local virtual environment.
- `src/models.py` silently fell back to `FallbackUNetPlusPlus`, a miniature 2-layer test network with 26 parameters.
- When `checkpoints/unet_plusplus_best.pth` (which contains 350 layers trained on ResNet34) was loaded with `strict=False`, exactly **0 out of 350 keys matched**. The fallback model was executing on uninitialized weights!
- **Resolution**: Installing `segmentation-models-pytorch` enabled `smp.UnetPlusPlus` and `smp.DeepLabV3Plus`. Loading `checkpoints/unet_plusplus_best.pth` and `deeplabv3_plus_best.pth` now achieves **100% strict key matching (`strict=True`)** with zero missing or unexpected keys.

---

## 4. Hardware Acceleration Performance on Apple M4

Benchmark testing on the local Apple M4 GPU via PyTorch Metal Performance Shaders (`mps`):
- **Batch Size**: 8 samples ($256 \times 256$ resolution)
- **Batches per Epoch**: 39 iterations (306 training tiles)
- **Time per Batch**: **$592.1\text{ ms}$**
- **Time per Full Epoch**: **$23.09\text{ seconds}$**
- **10-Epoch Training Estimate**: **$< 4\text{ minutes}$**

This enables rapid, local, reproducible training and hyperparameter tuning on the user's MacBook Air without incurring cloud GPU costs.
