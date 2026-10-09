# 📊 GeoVision AI — Satellite Water Body Segmentation Platform: Dataset Analysis Report

This report summarizes the details, checks, and spatial statistics of the remote sensing dataset compiled for the surface water body segmentation model.

---

## 📂 Dataset Split Statistics
*   **Total Imagery Pairs**: `414`
*   **Training Split (70% of train_val)**: `306` pairs
*   **Validation Split (15% of train_val)**: `54` pairs
*   **Test Split (Unbiased geographic scene)**: `54` pairs
*   **Segmented Patch Resolution**: `256 × 256 px`
*   **Color Profile**: RGB (3 channels BGR mapped)

---

## ⚖️ Class Balance & Frequencies
The pixel-level distribution between surface water and land/background pixels estimated from training masks:

| Land / Background (Class 0) | Surface Water (Class 1) |
| :--- | :--- |
| `96.42%` | `3.58%` |

> [!NOTE]
> Class imbalance is present in satellite data due to land masses dominant over small water basins. A custom hybrid **Dice-Focal Loss** is configured during training to handle this balance.

---

## 🧪 Data Quality & Validation Checks
*   **Verification 1: Shape Alignments**: Passed. Images and corresponding masks have identical spatial resolutions.
*   **Verification 2: Directory Checks**: Passed. Every training image path maps to an existing, non-empty ground-truth binary mask.
*   **Verification 3: Mask Scaling Bounds**: Passed. Values are checked programmatically and clamped to binary bounds `[0, 1]` upon loading.
