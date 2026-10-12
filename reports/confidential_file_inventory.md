# Confidential File & Valuable Artifact Inventory

**Audit Date**: October 12, 2026  
**Auditor**: Automated Production Security & Data Governance Review  
**Target Repository**: [`safalyareddy/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform`](https://github.com/safalyareddy/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform)  

---

## 1. Classification Categories Overview

Per security governance guidelines, every artifact is assigned one of six categories:
* **CONFIDENTIAL**: Sensitive private files that must never be exposed or committed.
* **SECRET FOUND**: Potential credential, key, or token requiring immediate investigation and redaction.
* **PRIVATE ARTIFACT**: Valuable heavy binary, dataset, or telemetry to be preserved securely; do not commit directly to Git.
* **LOCAL-ONLY SOURCE**: Source code currently missing from remote tracking that requires version control review.
* **SAFE TO COMMIT**: Appropriate documentation, configuration, manifests, or source code.
* **REVIEW REQUIRED**: Ambiguous artifact requiring explicit user direction.

---

## 2. Comprehensive Inventory Table

| Relative Path | Category | Exists Locally | Tracked by Git | Ignored by Git | In Target Remote | Potential Secret? | Important to Preserve? | Needs Backup? | Safe to Publish? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `checkpoints/unet_plusplus_best.pth` (100 MB) | **PRIVATE ARTIFACT** | YES | NO | YES | NO | NO | YES (Critical) | YES | CONDITIONAL (Model Hub / Release) |
| `checkpoints/unet_plusplus_best.onnx` (372 KB) | **PRIVATE ARTIFACT** | YES | NO | YES | NO | NO | YES (Critical) | YES | CONDITIONAL (Model Hub / Release) |
| `checkpoints/unet_plusplus_best.onnx.data` (99 MB) | **PRIVATE ARTIFACT** | YES | NO | YES | NO | NO | YES (Critical) | YES | CONDITIONAL (Model Hub / Release) |
| `checkpoints/deeplabv3_plus_best.pth` (86 MB) | **PRIVATE ARTIFACT** | YES | NO | YES | NO | NO | YES | YES | CONDITIONAL |
| `checkpoints/segformer_best.pth` (104 MB) | **PRIVATE ARTIFACT** | YES | NO | YES | NO | NO | YES | YES | CONDITIONAL |
| `checkpoints/*.ckpt` (Lightning 313 MB) | **PRIVATE ARTIFACT** | YES | NO | YES | NO | NO | YES | YES | NO (Raw training state) |
| `dataset/train/` (612 PNG tiles, ~20 MB) | **PRIVATE ARTIFACT** | YES | NO | YES | NO | NO | YES (Critical) | YES | CONDITIONAL (DVC / Open Portal) |
| `dataset/val/` (108 PNG tiles, ~4 MB) | **PRIVATE ARTIFACT** | YES | NO | YES | NO | NO | YES (Critical) | YES | CONDITIONAL (DVC / Open Portal) |
| `dataset/test/` (108 PNG tiles, ~4 MB) | **PRIVATE ARTIFACT** | YES | NO | YES | NO | NO | YES (Critical) | YES | CONDITIONAL (DVC / Open Portal) |
| `mlflow.db` (856 KB SQLite) | **PRIVATE ARTIFACT** | YES | NO | YES | NO | NO (0 secrets) | YES | YES | NO (Local telemetry DB) |
| `logs/tensorboard/version_0..10/*` (132 KB) | **PRIVATE ARTIFACT** | YES | NO | YES | NO | NO | YES | YES | NO (Telemetry curves) |
| `venv/` (358 files, 39 MB) | **PRIVATE ARTIFACT** | YES | NO | YES | NO | NO | NO | NO | NO (Platform binaries) |
| `.env`, `.env.local`, `.env.*` | **CONFIDENTIAL** | NO | NO | NO* | NO | NONE FOUND | N/A | N/A | NO (*Needs .gitignore entry) |
| `reports/dataset_split_manifest.json` | **SAFE TO COMMIT** | YES | YES | NO | NO | NO (Hashes only) | YES | YES | YES (Integrity manifest) |
| `benchmark_results.csv` | **SAFE TO COMMIT** | YES | YES | NO | YES | NO | YES | YES | YES (Evaluation benchmarks) |

---

## 3. High-Value Local Artifacts Summary

* **Checkpoints (Total 2.2 GB)**: Contains converged SMP UNet++ weights (Val Dice 0.8187), ONNX graphs with external tensor data, and legacy baseline weights. These represent hours of GPU training and must not be lost or deleted.
* **Dataset Tiles (Total 28 MB, 828 PNGs)**: Paired satellite imagery and binary masks across train, validation, and test splits. Re-downloading and re-tiling is time-consuming and introduces data drift risk.
* **Experiment History (`mlflow.db` & `logs/`)**: 11 recorded training runs and hyperparameter curves.
