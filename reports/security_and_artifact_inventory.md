# 🛡️ GeoVision AI — Security and Artifact Inventory Report (Phase 3)

**Date**: 2026-10-12  
**Auditor**: Senior Security Auditor & Machine Learning Engineer  
**Scope**: Secrets, Credentials, Local Databases, Checkpoints, Datasets, and Artifact Protection  

---

## 1. Security & Secrets Audit Findings

A thorough multi-vector scan was executed across the working directory, ignored folders, and entire Git history:

1. **Environment & Secret Files**:
   - `.env`, `.env.local`, `.pem`, `.key`, `id_rsa`: **NONE FOUND** (0 occurrences).
   - Git History Scan for `.env`: **CLEAN** (0 occurrences in entire historical DAG).
2. **Hardcoded Secrets & API Keys**:
   - Scanned regex pattern `(?i)(api_key|password|secret_key|private_key|token\s*=)`: **CLEAN** (0 occurrences).
3. **API Payload & Network Security**:
   - `src/api/main.py`: Enforces max payload limit ($\le 10\text{MB}$), restricts uploaded MIME types to valid image formats, and catches unhandled exceptions without leaking stack traces or internal server paths.
4. **Model Deserialization Safety**:
   - Production checkpoint loading in `src/api/main.py`, `src/evaluate.py`, and `app.py` enforces PyTorch `weights_only=True` to eliminate arbitrary code execution vulnerabilities during PyTorch unpickling.

---

## 2. Heavy & Irreplaceable Artifact Inventory

| Category | Artifact Path | Size | Description | Regeneration Feasibility | Git Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **Model Weights** | `checkpoints/best.ckpt` | 313.4 MB | PyTorch Lightning SegFormer checkpoint | High compute cost to regenerate | Ignored |
| **Model Weights** | `checkpoints/segformer_best.pth` | 104.5 MB | SegFormer-B2 PyTorch state dict | High compute cost to regenerate | Ignored |
| **Model Weights** | `checkpoints/unet_plusplus_best.pth`| 99.7 MB | UNet++ PyTorch state dict | High compute cost to regenerate | Ignored |
| **Model Weights** | `checkpoints/deeplabv3_plus_best.pth`| 85.8 MB| DeepLabV3+ PyTorch state dict | High compute cost to regenerate | Ignored |
| **ONNX Graphs** | `checkpoints/*.onnx` & `.onnx.data` | ~400 MB | ONNX Runtime traced inference models | Deterministically exportable | Ignored |
| **Tracking DB** | `mlflow.db` | 876.5 KB | SQLite database containing MLflow experiment runs | Irreplaceable experiment history | Ignored |
| **Experiment Logs**| `logs/tensorboard/` | 16.0 KB | TensorBoard event logs | Irreplaceable historical loss curves | Ignored |
| **Satellite Data** | `dataset/train/` | ~21.0 MB | 306 paired 256x256 Sentinel-2 tiles | Irreplaceable ground-truth masks | Ignored |
| **Satellite Data** | `dataset/val/` | ~3.5 MB | 54 paired 256x256 Sentinel-2 tiles | Irreplaceable ground-truth masks | Ignored |
| **Satellite Data** | `dataset/test/` | ~3.5 MB | 54 held-out test Sentinel-2 tiles | Irreplaceable ground-truth masks | Ignored |
| **Virtual Env** | `venv/` | ~4.8 GB | Python 3.11 virtual environment | Regenerable via `pip install` | Ignored |

---

## 3. Licensing & Data Provenance

- **Imagery Source**: European Space Agency (ESA) Sentinel-2 satellite missions.
- **Licensing**: Open Access under Copernicus Sentinel Data Policy (Free, full and open access for non-commercial and commercial use with attribution).
- **Public Distribution Policy**: Safe for portfolio and research demonstration. Model weights can be hosted via Hugging Face Hub or Git LFS release assets rather than committing directly to Git trees.
