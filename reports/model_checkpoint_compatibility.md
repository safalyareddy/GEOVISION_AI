# GeoVision AI — Model Checkpoint Compatibility & Deserialization Audit

**Audit Date**: October 13, 2026  
**Auditor**: Senior Machine Learning Systems & QA Engineer  
**Hardware Platform**: Apple Silicon MacBook Air M4 (MPS & CPU Execution)  

---

## 1. Checkpoint Inventory & Integrity Summary

Every model checkpoint present in `checkpoints/` was audited for cryptographic integrity, safe deserialization compatibility (`weights_only=True`), layer key prefixes, and strict state dictionary matching against target neural architectures.

| Checkpoint File | Size | Cryptographic SHA-256 | Safe Load (`weights_only=True`) | Target Architecture | Strict Match (`strict=True`) | Audit Verdict |
| :--- | :---: | :--- | :---: | :--- | :---: | :--- |
| **`unet_plusplus_best.pth`** | 99.68 MB | `e9ac1bed148e6f3e8926d9008fd3fe6ad490ff8ed1dee6d79b96134ef8712c9b` | **PASS** | SMP UNet++ (ResNet-34) | **0 missing / 0 unexpected** | **PRODUCTION READY** |
| **`unet_plusplus_trained.pth`** | 99.68 MB | `994b8a28655cd09e1a9fe6dbda1b871bed194604d006ab92a55f6923092532d3` | **PASS** | SMP UNet++ (ResNet-34) | **0 missing / 0 unexpected** | **VALID CONVERGED CHECKPOINT** |
| **`deeplabv3_plus_best.pth`** | 85.76 MB | `58efb3a365479b99483bfc014c301050306456747396262ef1bec3ea7646201b` | **PASS** | SMP DeepLabV3+ (ResNet-34) | **0 missing / 0 unexpected** | **BENCHMARK COMPATIBLE** |
| **`segformer_best.pth`** | 104.48 MB | `5b55c75750f8680ab3c6c9819e0f3da46a35b17d65863e54cca07b7816863749` | **PASS** | SegFormer-B2 (MiT-B2) | **0 missing / 0 unexpected** | **BENCHMARK COMPATIBLE** |
| `best_model.pth` | 189.51 MB | `3047dfe5a48214a3f3fc7e5d4a0b67c415dd674cfb66fc274d82c7f548934680` | **FAIL** (Miniz archive error) | Unknown Legacy Architecture | N/A | **CORRUPT ARCHIVE (ARCHIVED)** |
| `best.ckpt` | 313.42 MB | `a61cc48a65878684f1b61c16a2fa89b1c4cf2e70adca7ed651117fa02fad3166` | **FAIL** (Module missing) | Legacy PyTorch Lightning | N/A | **LEGACY DEPENDENCY (`src.config`)** |
| `last.ckpt` | 313.42 MB | `a61cc48a65878684f1b61c16a2fa89b1c4cf2e70adca7ed651117fa02fad3166` | **FAIL** (Module missing) | Legacy PyTorch Lightning | N/A | **LEGACY DEPENDENCY (`src.config`)** |
| `segformer_best.ckpt` | 313.42 MB | `a61cc48a65878684f1b61c16a2fa89b1c4cf2e70adca7ed651117fa02fad3166` | **FAIL** (Module missing) | Legacy PyTorch Lightning | N/A | **LEGACY DEPENDENCY (`src.config`)** |
| `segformer_last.ckpt` | 313.42 MB | `a61cc48a65878684f1b61c16a2fa89b1c4cf2e70adca7ed651117fa02fad3166` | **FAIL** (Module missing) | Legacy PyTorch Lightning | N/A | **LEGACY DEPENDENCY (`src.config`)** |

---

## 2. Investigation of Non-Strict Loading (`strict=False`)

In prior audit passes, evaluation scripts (`src/evaluate.py`) utilized `model.load_state_dict(state, strict=False)`. This investigation tested whether models were silently misconfigured:

1. **SMP UNet++**:
   - Total model parameters: `26,078,609` (176 trainable weight/bias tensors + 174 BatchNorm running mean/variance buffers = 350 state tensors).
   - State dictionary in `unet_plusplus_best.pth` contains exactly 350 tensors.
   - Calling `model.load_state_dict(state, strict=True)` succeeds with **0 missing keys and 0 unexpected keys**.
2. **SMP DeepLabV3+**:
   - Total state tensors: 277 tensors.
   - Calling `model.load_state_dict(state, strict=True)` succeeds with **0 missing keys and 0 unexpected keys**.
3. **SegFormer-B2 (`HuggingFaceSegFormerWrapper`)**:
   - Total state tensors: 380 tensors.
   - In `HuggingFaceSegFormerWrapper`, the sub-module is `self.model`, so parameters have the prefix `model.segformer...`.
   - The checkpoint `segformer_best.pth` contains identical `model.segformer...` keys.
   - Calling `model.load_state_dict(state, strict=True)` succeeds with **0 missing keys and 0 unexpected keys**.
4. **Root Cause Conclusion**:
   The use of `strict=False` in earlier evaluation code was generic defensive programming rather than an indicator of architectural mismatch or missing weights. Both `src/evaluate.py` and `src/api/main.py` have now been updated to attempt `strict=True` first, guaranteeing that silent fallback or weight corruption cannot occur unnoticed.

---

## 3. Mathematical Verification of Metric Formulas

Evaluation formulas were independently verified against hand-computable test matrices:

$$\text{Dice} = \frac{2 \times \text{TP} + \epsilon}{2 \times \text{TP} + \text{FP} + \text{FN} + \epsilon}, \quad \text{IoU} = \frac{\text{TP} + \epsilon}{\text{TP} + \text{FP} + \text{FN} + \epsilon}$$

### Test Matrix 1: Hand-Computable Partial Overlap ($2 \times 2$)
* Ground Truth: $\begin{bmatrix} 1 & 0 \\ 1 & 0 \end{bmatrix}$ (2 positive, 2 negative)
* Model Prediction: $\begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}$ (2 positive, 2 negative)
* Confusion Matrix: $\text{TP}=1, \text{FP}=1, \text{FN}=1, \text{TN}=1$
* Expected Metrics:
  - $\text{Dice} = \frac{2(1)}{2(1) + 1 + 1} = \frac{2}{4} = \mathbf{0.5000}$
  - $\text{IoU} = \frac{1}{1 + 1 + 1} = \frac{1}{3} = \mathbf{0.3333}$
  - $\text{Precision} = \frac{1}{1 + 1} = \mathbf{0.5000}, \quad \text{Recall} = \frac{1}{1 + 1} = \mathbf{0.5000}$
* **Automated Unit Test**: Verified in [`tests/test_evaluate.py:test_calculate_metrics_hand_computable_partial_overlap`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/tests/test_evaluate.py#L35) (**PASSED**).

### Test Matrix 2: Empty Mask True Negative
* Ground Truth: All zeros ($4 \times 4$), Prediction: All zeros ($4 \times 4$)
* $\text{TP}=0, \text{FP}=0, \text{FN}=0$. With Laplace smoothing ($\epsilon = 10^{-6}$), $\text{Dice} = \frac{10^{-6}}{10^{-6}} = \mathbf{1.0000}$.
* **Automated Unit Test**: Verified in [`tests/test_evaluate.py:test_calculate_metrics_empty_mask_handling`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/tests/test_evaluate.py#L52) (**PASSED**).
