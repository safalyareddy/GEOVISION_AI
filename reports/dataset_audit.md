# 🔬 GeoVision AI — Dataset Forensic Audit Report (Phase 4)

**Date**: 2026-10-12  
**Auditor**: Senior Computer Vision Researcher & Machine Learning Engineer  
**Dataset**: Sentinel-2 Surface Water Body Segmentation Dataset  

---

## 1. Dataset Provenance, Attribution & Specifications

- **Imagery Origin**: European Space Agency (ESA) Copernicus Sentinel-2 Level-2A surface reflectance data.
- **Sensor Specs**: Multispectral Instrument (MSI), True Color RGB visual composite bands (B04-Red, B03-Green, B02-Blue).
- **Ground Spatial Resolution**: 10 meters per pixel ($1 \text{ pixel} = 100\text{ m}^2$; $256 \times 256 \text{ tile} = 6.55 \text{ km}^2$).
- **License**: Open Access under Copernicus Sentinel Data Policy (Free and open with attribution).
- **Mask Encoding**: 8-bit grayscale images with discrete binary values:
  - $\mathbf{0}$ = Background / Land / Vegetation / Built-up area.
  - $\mathbf{255}$ = Surface Water body (rivers, lakes, coastal margins, wetlands).

---

## 2. Empirical Dataset Distribution & Partitioning

Forensic evaluation was executed across every individual sample in `dataset/`:

| Split Partition | Sample Count | Total Pixels | Water Pixels | Land Pixels | Water % | Empty Masks (0% water) | All-Water Masks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Split** | 306 | 20,054,016 | 846,471 | 19,207,545 | **4.22%** | 28 | 0 |
| **Validation Split**| 54 | 3,538,944 | 240,432 | 3,298,512 | **6.79%** | 3 | 0 |
| **Held-Out Test** | 54 | 3,538,944 | 361,603 | 3,177,341 | **10.22%** | 3 | 0 |
| **TOTAL DATASET** | **414** | **27,138,952** | **1,448,506** | **25,690,446** | **5.34%** | **34** | **0** |

### Key Forensic Findings:
1. **Actual vs Documented Class Imbalance**:
   - The README historically cited a `3.58% water / 96.42% land` distribution.
   - The actual current dataset has **5.34% surface water vs. 94.66% land**.
   - This still represents an extreme $\sim 18:1$ class imbalance, confirming that standard Binary Cross-Entropy (BCE) loss is inappropriate and validating the necessity of **Hybrid Dice-Focal Loss**.
2. **Hard Negative Samples (Empty Masks)**:
   - 34 tiles (8.21% of the dataset) contain zero water pixels. These pure-land tiles are crucial hard negatives that prevent false positive predictions on forests, shadows, and asphalt.
3. **Data Leakage & Integrity**:
   - **Cross-Split Leakage**: **0 occurrences** (MD5 hash analysis confirmed no duplicate or identical image tiles shared across train, val, or test splits).
   - **Corrupted / Invalid Samples**: **0 occurrences** (All 414 images and masks opened without error, dimension match is 100% $256 \times 256$).
   - **Unmatched Image-Mask Pairs**: **0 occurrences**.

---

## 3. Preprocessing, Augmentation & Normalization Audit

1. **Preprocessing Parity**:
   - **Training Preprocessing** (`src/data/dataset.py:192-235`):
     - Images resized to $256 \times 256$, converted to FloatTensor, scaled to $[0.0, 1.0]$.
     - Masks loaded in `L` mode, converted to FloatTensor $\{0.0, 1.0\}$.
   - **Inference Preprocessing** (`src/api/main.py:133-145` & `app.py:82-95`):
     - Identical: $256 \times 256$ resize, FloatTensor normalization $[0.0, 1.0]$.
2. **Channel Dimensions**:
   - Input tensors are $[B, 3, 256, 256]$ matching standard RGB Sentinel-2 composites.
   - Masks are $[B, 1, 256, 256]$.
3. **Artifacts Generated**:
   - `reports/dataset_statistics.json` (Full numerical dataset statistics).
   - `reports/dataset_split_manifest.json` (Per-sample manifest with dimensions, water fraction, and MD5 hashes).
   - `reports/dataset_issues.csv` (0 issues logged).
