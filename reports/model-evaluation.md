# GeoVision AI — Authoritative Model Evaluation & Benchmark Report

This document is the verified, authoritative record of model evaluation for the GeoVision AI satellite water body segmentation platform. It reports the independent, reproducible evaluation of both baseline models and improved models on the strictly held-out test dataset.

---

## 1. Executive Summary & Verified Findings

- **Test Split Context**: 54 held-out Sentinel-2 satellite image patches ($256 \times 256$ pixels), totaling **3,538,944 pixels**.
- **Extreme Class Imbalance**: Ground truth in the held-out test split contains **473,646 water pixels (13.38%)** and **3,065,298 land/background pixels (86.62%)**. Across dataset splits, surface water coverage is: **3.89%** in train (779,844 / 20,054,016 px), **5.51%** in validation (194,823 / 3,538,944 px), **4.13%** across the train+validation pool (974,667 / 23,592,960 px), and **5.34%** overall across all 414 tiles combined (1,448,313 / 27,131,904 px, a 17.7:1 land-to-water ratio).
- **All-Background Baseline Accuracy**: A trivial classifier predicting 100% background (land) achieves **86.62% Pixel Accuracy** ($3,065,298 / 3,538,944$) on the held-out test split while capturing **0.0% Water Recall**, demonstrating why pixel accuracy alone is misleading for satellite water detection.
- **The Core Bottleneck**: Baseline models trained with standard symmetric Dice / BCE loss suffered from high pixel accuracy (87%–89%) but severe false negatives (water recall as low as 8.7% for SegFormer and 29.8% for UNet++).
- **The Solution**: Formulating asymmetric recall-prioritized loss functions (**Tversky Loss** with $\beta=0.7 > \alpha=0.3$ and **Focal Tversky Loss** with $\gamma=1.33$) combined with multi-scale architecture (**Feature Pyramid Network (FPN) with ResNet-34**).
- **Champion Model**: **Improved FPN (EXP06)** achieved **54.24% Water Recall** (+82.3% relative improvement over baseline UNet++: $29.76\% \to 54.24\%$, +24.48 percentage points), cutting false negative water pixels from 332,710 to 216,760 (**34.8% false-negative reduction**, 115,950 false negative pixels recovered), while improving **Pixel Accuracy to 91.56%**, **Water Precision to 75.81%**, and **Micro Dice to 0.6323** (Macro Dice 0.4580).

---

## 2. Authoritative Benchmark Comparison Table

All metrics below are computed on the identical 54 held-out test images at frozen threshold $\tau = 0.50$ (with calibrated $\tau^* = 0.60$ where applicable). Water is treated as the positive class ($1$).

| Model Architecture & Experiment | Model Status | Threshold ($\tau$) | Pixel Accuracy (%) | Water Precision (%) | Water Recall (%) | Macro Dice | Macro IoU | Micro Dice | Micro IoU | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNet++ (Exp1 Baseline)** | Baseline | 0.50 | 89.38% | 76.55% | 29.76% | 0.3032 | 0.2035 | 0.4285 | 0.2727 | 140,936 | 43,163 | 332,710 | 3,022,135 |
| **DeepLabV3+ (Exp2 Baseline)** | Baseline | 0.50 | 89.47% | 89.70% | 24.07% | 0.2211 | 0.1464 | 0.3795 | 0.2342 | 113,995 | 13,091 | 359,651 | 3,052,207 |
| **SegFormer-B0 (Exp3 Baseline)** | Baseline | 0.50 | 87.59% | 85.84% | 8.74% | 0.1083 | 0.0615 | 0.1586 | 0.0861 | 41,385 | 6,828 | 432,261 | 3,058,470 |
| **Improved UNet++ (EXP01 Tversky)** | Improved | 0.50 | 90.64% | 80.37% | 39.75% | 0.3872 | 0.2718 | 0.5319 | 0.3623 | 188,256 | 45,992 | 285,390 | 3,019,306 |
| **Improved UNet++ (EXP01 Calibrated)** | Calibrated | 0.60 | 90.60% | 80.76% | 39.12% | 0.3835 | 0.2686 | 0.5271 | 0.3578 | 185,289 | 44,151 | 288,357 | 3,021,147 |
| **Improved UNet++ (EXP02 Focal Tversky)** | Improved | 0.50 | 90.05% | 76.60% | 36.89% | 0.3367 | 0.2339 | 0.4980 | 0.3316 | 174,737 | 53,381 | 298,909 | 3,011,917 |
| **Improved DeepLabV3+ (EXP04 Focal Tversky)** | Improved | 0.50 | 90.22% | 78.57% | 37.05% | 0.3636 | 0.2542 | 0.5036 | 0.3365 | 175,501 | 47,872 | 298,145 | 3,017,426 |
| **Improved DeepLabV3+ (EXP04 Calibrated)** | Calibrated | 0.60 | 90.15% | 79.52% | 35.54% | 0.3542 | 0.2466 | 0.4913 | 0.3256 | 168,339 | 43,347 | 305,307 | 3,021,951 |
| **Improved SegFormer-B0 (EXP05 Focal Tversky)** | Improved | 0.50 | 87.03% | 52.51% | 32.75% | 0.3414 | 0.2275 | 0.4034 | 0.2527 | 155,141 | 140,320 | 318,505 | 2,924,978 |
| **Improved FPN (EXP06 Focal Tversky) — CHAMPION** | **Champion** | **0.50** | **91.56%** | **75.81%** | **54.24%** | **0.4580** | **0.3328** | **0.6323** | **0.4624** | **256,886** | **81,954** | **216,760** | **2,983,344** |
| **Improved FPN (EXP06 Calibrated)** | Calibrated | 0.60 | 91.56% | 76.45% | 53.35% | 0.4544 | 0.3295 | 0.6284 | 0.4582 | 252,677 | 77,835 | 220,969 | 2,987,463 |

---

## 3. Metric Definitions & Distinctions

To ensure complete clarity:

1. **Pixel Accuracy**: $\frac{TP + TN}{TP + FP + FN + TN}$.
   * *Caveat*: In highly imbalanced datasets, predicting all background yields high accuracy (~87%) while failing completely as a detector. Pixel accuracy must always be interpreted alongside Recall and Dice.
2. **Water Precision**: $\frac{TP}{TP + FP}$. Fraction of predicted water pixels that are genuinely water.
3. **Water Recall**: $\frac{TP}{TP + FN}$. Fraction of true water pixels correctly captured by the model.
4. **Macro vs. Micro Averaging**:
   - **Micro Dice**: Computed globally over all accumulated pixels across the entire test split:
     $$\text{Dice}_{\text{micro}} = \frac{2 \cdot \sum TP}{2 \cdot \sum TP + \sum FP + \sum FN}$$
   - **Macro Dice**: Unweighted arithmetic mean of per-image Dice scores across all 54 test images:
     $$\text{Dice}_{\text{macro}} = \frac{1}{N} \sum_{i=1}^N \text{Dice}_i$$
     *Images with zero ground-truth water pixels that have zero false positives receive a Dice of 1.0; if false positives occur, Dice is 0.0.*
5. **Macro vs. Micro IoU**:
   - **Micro IoU**: $\frac{\sum TP}{\sum TP + \sum FP + \sum FN}$
   - **Macro IoU**: $\frac{1}{N} \sum_{i=1}^N \text{IoU}_i$

---

## 4. Evaluation Rigor & Verification Protocols

1. **Zero Data Leakage**:
   - Train (306 images), validation (54 images), and test (54 images) splits were generated by `scripts/split_dataset.py` with seed 42.
   - All evaluation scripts load strictly from `dataset/test/images/` and `dataset/test/masks/`.
   - No test images or masks were observed during training, loss computation, or threshold selection.
2. **Deterministic Preprocessing**:
   - Optical images loaded as 3-channel RGB float tensors normalized to $[0, 1]$ across all models (FPN, UNet++, DeepLabV3+, SegFormer), with identical preprocessing applied during training and evaluation.
   - Masks loaded as single-channel binary float tensors ($0.0$ or $1.0$).
3. **Validation-Only Decision Threshold Calibration**:
   - Default benchmark: $\tau = 0.50$ (standard frozen threshold across all comparisons).
   - Optimal validation threshold $\tau^* = 0.60$ was discovered via validation sweep ($0.10 \le \tau \le 0.90$ in steps of 0.05) conducted exclusively on the 54-patch validation split.
   - The calibrated threshold $\tau^* = 0.60$ was then evaluated frozen on the 54 held-out test patches to confirm out-of-sample generalization, proving that gains are not an artifact of test-set threshold tuning.
4. **Hardware Environment**:
   - Evaluated on Apple Silicon M4 with PyTorch 2.14.1 on MPS / CPU runtime.

---

## 5. Architectural Findings: Why FPN Won

1. **Multi-Scale Feature Hierarchy**:
   - Satellite water bodies range from massive coastal oceans to 1-pixel-wide irrigation canals and narrow streams.
   - Standard decoders (DeepLab ASPP, UNet++) struggle with fine boundary details when feature maps are downsampled $16\times$ or $32\times$.
   - FPN's lateral connections directly inject high-resolution feature maps ($P_2, P_3$) into the top-down pyramid, preserving boundary fidelity.
2. **Loss Function Synergies**:
   - Symmetric Dice loss treats false positives and false negatives equally. Under severe class imbalance, suppressing false positives is mathematically easier than recovering false negatives.
   - Focal Tversky Loss ($\alpha=0.3, \beta=0.7, \gamma=1.33$) penalizes false negatives $2.33\times$ more heavily than false positives, forcing the optimizer to segment thin water bodies.

---

## 6. Real-World Limitations & Scope of Claims

1. **Geographic Scope**:
   - The dataset consists of Sentinel-2 optical tiles over specific geographical regions.
   - **No universal geographic generalization is claimed**. High accuracy on this test set does not imply equal performance over glaciated terrain, sand dunes, or heavy cloud cover without domain adaptation.
2. **Sensor Constraints**:
   - Optical RGB imagery is subject to cloud occlusion, sun glint, and turbid water confusion (shallow mudflats).
   - Multi-spectral bands (NIR, SWIR for MNDWI computation) or SAR (Sentinel-1) would be necessary for all-weather operational remote sensing.
3. **Physical Area Approximations**:
   - Water area ($\text{km}^2$) calculated by the API/UI assumes nominal 10m Ground Sample Distance ($100 \text{ m}^2/\text{pixel}$). When geospatial CRS metadata is absent (standard PNG inputs), this serves as an unreferenced approximation.
