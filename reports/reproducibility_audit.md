# Phase Thirteen — Reproducibility & Scientific Provenance Audit

**Audit Date**: October 12, 2026  
**Auditor**: Lead MLOps Engineer & Research Scientist  
**Provenance Manifest**: [`reports/dataset_split_manifest.json`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/reports/dataset_split_manifest.json)  

---

## 1. Determinism Controls & Seed Management

* **Global Random Seed**: Fixed at seed `42` across Python `random`, `numpy.random`, and `torch.manual_seed`.
* **Deterministic Splitting**: `scripts/split_dataset.py` uses fixed seeding and stratified class balancing to ensure that re-partitioning the raw tiles produces identical train (306), val (54), and test (54) splits.
* **Deterministic Inference Transforms**: Preprocessing during evaluation and inference uses deterministic Albumentations / TorchVision operators (bilinear interpolation and standard z-score normalization), guaranteeing zero stochastic variation during testing.

---

## 2. Checkpoint Provenance & Configuration Isolation

1. **Hydra Configuration Management**:
   * Architecture, optimizer hyperparameters, learning rates, and dataset paths are decoupled into YAML files under `config/`.
2. **Hardware Environment Snapshot**:
   * Hardware: Apple Silicon M4, macOS 26.5.1 Darwin arm64.
   * Software: Python 3.11.15, PyTorch 2.14.1, ONNX 1.16+, TorchVision 0.19+.
3. **Artifact Verification**:
   * MD5 and SHA-256 hash manifests recorded in `reports/dataset_split_manifest.json` and `reports/secure_backup_checklist.md`.
