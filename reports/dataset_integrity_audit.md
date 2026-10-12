# Phase Two — Dataset Integrity & Forensics Audit

**Audit Date**: October 12, 2026  
**Auditor**: Lead Data Engineer & Computer Vision Researcher  
**Dataset Directory**: `/Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/dataset`  

---

## 1. Partition Verification & Forensics Summary

The dataset on disk was empirically audited tile-by-tile across all three partitions:

| Metric | Train Partition | Validation Partition | Test Partition (Held-Out) | Total / Combined |
| :--- | :---: | :---: | :---: | :---: |
| **Paired Image-Mask Tiles** | **306** | **54** | **54** | **414** |
| **Resolution & Channels** | $256 \times 256 \times 3$ RGB | $256 \times 256 \times 3$ RGB | $256 \times 256 \times 3$ RGB | Uniform $256 \times 256$ |
| **Mask Encoding & Shape** | $256 \times 256 \times 1$ uint8 | $256 \times 256 \times 1$ uint8 | $256 \times 256 \times 1$ uint8 | Binary $\{0, 255\}$ |
| **1-to-1 Pair Matching** | **100% (306/306)** | **100% (54/54)** | **100% (54/54)** | **100% Match** |
| **Water Pixel Ratio** | **3.89%** | **5.51%** | **13.38%** | **5.34% Average** |
| **Background (Land) Ratio**| **96.11%** | **94.49%** | **86.62%** | **94.66% Average** |
| **Empty Masks (Hard Negatives)** | **27 (8.8%)** | **7 (13.0%)** | **0 (0.0%)** | **34 (8.2%)** |
| **Corrupted / Truncated Files** | **0** | **0** | **0** | **0** |
| **Duplicate Hashes Across Splits**| **0 (No Leakage)** | **0 (No Leakage)** | **0 (No Leakage)** | **0 Overlap** |

---

## 2. Integrity Manifest & Leakage Verification

* **Cryptographic MD5 Hash Verification**: Every tile in `dataset/train/`, `dataset/val/`, and `dataset/test/` has an immutable entry recorded in `reports/dataset_split_manifest.json`.
* **Zero Cross-Split Contamination**: Intersection analysis of image MD5 hashes confirms that train, validation, and test splits are mutually disjoint:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
* **Hard Negative Mining**: The training set deliberately contains 27 negative tiles (0% water pixels) to teach the network to suppress false positives on barren land, arid regions, and urban infrastructure.

---

## 3. Geospatial Metadata & Physical Area Limitations

* **Observation**: Current tiles are stored as standard 8-bit RGB PNG files derived from Sentinel-2 optical imagery rather than raw GeoTIFF format with embedded coordinate reference systems (CRS) and bounding boxes.
* **Calibrated Ground Sampling Distance (GSD)**: Under Sentinel-2 optical resolution (10 meters/pixel):
  $$\text{Pixel Area} = 10\text{ m} \times 10\text{ m} = 100\text{ m}^2 = 0.01\text{ hectares}$$
  $$\text{Tile Area} = 256 \times 256 \times 100\text{ m}^2 = 6,553,600\text{ m}^2 \approx 6.55\text{ km}^2$$
* **Scientific Boundary**: Because PNG headers discard exact UTM coordinates and orbital sun angles, physical area figures in the Streamlit UI and API responses are labeled as **calibrated estimates assuming nominal 10m GSD**, rather than certified survey measurements.
