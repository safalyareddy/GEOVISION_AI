# Dataset Provenance, Forensics & Ground Truth

---

## 1. Dataset Overview

GeoVision AI is evaluated on high-resolution Sentinel-2 optical imagery paired with hand-annotated and multi-spectral validated binary water masks.

* **Sensor**: Sentinel-2 Multi-Spectral Instrument (MSI) Level-1C / Level-2A.
* **Resolution**: Nominal Ground Sampling Distance (GSD) of 10 meters per pixel.
* **Tile Dimensions**: $256 \times 256$ pixels (approx. $2.56 \text{ km} \times 2.56 \text{ km} = 6.55 \text{ km}^2$ per patch).
* **Total Sample Count**: 414 paired image-mask tiles.

---

## 2. Partition Breakdown & Class Distributions

| Split | Tile Count | Percentage | Water Pixels | Land Pixels | Hard Negatives (0% Water) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 306 | 73.9% | 3.89% | 96.11% | 27 tiles (8.8%) |
| **Validation** | 54 | 13.0% | 5.51% | 94.49% | 7 tiles (13.0%) |
| **Test (Held-Out)** | 54 | 13.0% | 13.38% | 86.62% | 0 tiles (0.0%) |
| **Total** | **414** | **100%** | **5.34%** | **94.66%** | **34 tiles (8.2%)** |

---

## 3. Provenance & Attribution

* **Source Imagery**: European Space Agency (ESA) Copernicus Sentinel-2 Open Access Data.
* **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) / Open Access for academic research and application development.
* **Reproducibility**: Exact partition hashes recorded in `reports/dataset_split_manifest.json`.
