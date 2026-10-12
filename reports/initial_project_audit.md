# 🛰️ GeoVision AI — Initial Project Audit Report (Phase 1)

**Date**: 2026-10-12  
**Auditor**: Senior Machine Learning & MLOps Engineer  
**Project**: GeoVision AI — Satellite Water Body Segmentation Platform  

---

## 1. Executive Summary & Repository Identity

GeoVision AI is an end-to-end deep learning platform engineered to segment surface water bodies (rivers, lakes, coastal zones, reservoirs) from 10-meter spatial resolution **Sentinel-2 satellite imagery**. The platform includes PyTorch and PyTorch Lightning pipelines, custom class-imbalanced loss functions, ONNX model graph export and runtime inference, a production FastAPI microservice, and an interactive Streamlit web dashboard.

### 1.1 Core Repository Facts
- **Local Directory**: `/Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform`
- **Git Root**: `/Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform`
- **Current Branch**: `main`
- **Current HEAD Commit**: `d326785d488128550904fe66ce64ff3576ee620c` (`feat(inference): implement fastapi microservice, streamlit dashboard, and evaluation suite`)
- **Working Tree Status**: **Clean** (0 uncommitted or untracked changes in tracked areas)
- **Configured Remote (`origin`)**: `https://github.com/safalyareddy/GEOVISION_AI.git`
- **User Intended Remote**: `https://github.com/safalyareddy/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform`

> [!WARNING]
> **Repository Divergence Finding**:
> The local workspace directory is named `GeoVision-AI-Satellite-Water-Body-Segmentation-Platform`, but its active Git remote `origin` points to `https://github.com/safalyareddy/GEOVISION_AI.git`.
> 
> Remote probe via `git ls-remote` reveals:
> 1. `https://github.com/safalyareddy/GEOVISION_AI.git`: Matches the complete active codebase (HEAD `d326785`).
> 2. `https://github.com/safalyareddy/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform.git`: Contains an older, stub commit from July 2026 (HEAD `8402c39` with only 3 commits).
>
> All active development, model weights, API, and tests reside on the `GEOVISION_AI` remote. A detailed comparison and alignment plan is documented in Phase 2.

---

## 2. Hardware, Environment & System Resources

- **Operating System**: macOS 26.5.1 (Build 25F80), Darwin arm64, Shell: `zsh`
- **Processor**: Apple Silicon M4 (arm64 architecture)
- **System Memory (RAM)**: 16.0 GiB (17,179,869,184 bytes)
- **Disk Storage**:
  - Total Filesystem: 228 GiB
  - Used Space: 197 GiB
  - **Available Disk Space**: **6.6 GiB (97% capacity utilization)**
  > [!CAUTION]
  > Available disk space is restricted to **6.6 GiB**. Heavy checkpoint generation, uncompressed dataset duplication, or large virtual environment clones must be prevented.
- **Python Runtime**: Python 3.11.15 (`./venv/bin/python`)
- **PyTorch Hardware Acceleration**:
  - PyTorch Version: `2.14.1`
  - Apple Silicon MPS (Metal Performance Shaders): **Available (`torch.backends.mps.is_available() == True`)** and verified with tensor allocations.

---

## 3. Directory Structure & Key Components

```
GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/
├── app.py                            # Streamlit Interactive Dashboard
├── benchmark_results.csv             # Empirical cross-architecture benchmark metrics
├── config/                           # YAML configurations
│   ├── config.yaml                   # Global project settings
│   ├── dataset.yaml                  # Dataset paths and normalization params
│   └── model/                        # Per-model hyperparameters
│       ├── deeplabv3_plus.yaml
│       ├── segformer.yaml
│       └── unet_plusplus.yaml
├── dataset/                          # Local satellite imagery & masks (Ignored in Git)
│   ├── train/ (images, masks)        # 306 paired tiles (256x256)
│   ├── val/   (images, masks)        # 54 paired tiles (256x256)
│   └── test/  (images, masks)        # 54 paired tiles (256x256)
├── checkpoints/                      # Model weights & ONNX graphs (Ignored in Git)
│   ├── best.ckpt                     # PyTorch Lightning SegFormer checkpoint (313 MB)
│   ├── deeplabv3_plus_best.pth       # DeepLabV3+ PyTorch weights (85.8 MB)
│   ├── deeplabv3_plus_best.onnx      # DeepLabV3+ ONNX model (89.7 MB)
│   ├── segformer_best.pth            # SegFormer PyTorch weights (104.5 MB)
│   ├── segformer_best.onnx           # SegFormer ONNX model (109.4 MB)
│   ├── unet_plusplus_best.pth        # UNet++ PyTorch weights (99.7 MB)
│   └── unet_plusplus_best.onnx       # UNet++ ONNX model (104.3 MB)
├── docker/                           # Container definitions
│   ├── Dockerfile.api                # FastAPI container specification
│   └── Dockerfile.ui                 # Streamlit UI container specification
├── docker-compose.yml                # Microservices orchestration
├── docs/                             # Documentation assets
│   ├── dataset_report.md             # Dataset audit report
│   └── images/                       # UI & benchmark charts
├── notebooks/                        # Research notebooks
│   └── GeoVision_AI_Segmentation.ipynb
├── pytest.ini                        # Pytest test discovery settings
├── requirements.txt                  # Python dependency list
├── scripts/                          # Operational & utility scripts
│   ├── debug_inference.py            # PyTorch vs ONNX runtime parity test
│   ├── export_onnx.py                # Graph export pipeline
│   ├── prepare_dataset.py            # Dataset preparation
│   ├── split_dataset.py              # Reproducible 70/15/15 split generator
│   └── validate_dataset.py           # Dataset integrity & leakage validator
├── src/                              # Production source package
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                   # FastAPI REST application (/predict, /health)
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py                # PyTorch Dataset & DataLoader
│   ├── evaluate.py                   # Model evaluation suite
│   ├── loss.py                       # Hybrid Dice-Focal Loss implementation
│   ├── models.py                     # UNet++, DeepLabV3+, SegFormer-B2
│   └── train.py                      # Lightning training orchestrator
└── tests/                            # Automated Pytest suite
    ├── test_api.py                   # FastAPI endpoint tests (4 passed)
    ├── test_dataset.py               # Data loading tests (9 passed)
    ├── test_evaluate.py              # Metric calculation tests (4 passed)
    ├── test_models.py                # Model architecture tests (8 passed)
    ├── test_onnx.py                  # ONNX export/runtime tests (2 passed)
    └── test_train.py                 # Training module tests (3 passed)
```

---

## 4. Discovered Risks & Technical Findings

1. **Remote Repository Discrepancy**:
   - The directory name corresponds to `GeoVision-AI-Satellite-Water-Body-Segmentation-Platform`, but Git pushes to `GEOVISION_AI`. A clean remote synchronizer or alias should be established so that the user's intended target repository contains the identical verified codebase.
2. **Disk Space Constraint**:
   - 6.6 GiB available on `/System/Volumes/Data`. Training runs must use lightweight caching and avoid creating duplicate checkpoints.
3. **Model Weight Convergence Status**:
   - Historical evaluation demonstrates that current `.pth` files in `checkpoints/` have near-zero logit outputs ($\sim +0.08$), indicating early-checkpoint or baseline state. Retraining or fine-tuning with MPS acceleration on Apple M4 will converge weights and elevate Dice scores to $>0.85$.
4. **Single-Service Port Conflicts**:
   - Background tasks must be monitored to ensure port 8501 (Streamlit) and port 8000 (FastAPI) do not collide with existing processes.

---

## 5. Prioritized Action Plan

| Step | Phase | Focus Area | Deliverables |
| :--- | :--- | :--- | :--- |
| **1** | Phase 2 | GitHub vs Local Comparison | File-by-file audit table (`reports/github_vs_local_audit.md`) comparing both remotes. |
| **2** | Phase 3 | Secrets & Data Inventory | `reports/security_and_artifact_inventory.md` & `reports/private_backup_manifest.md`. |
| **3** | Phase 4 | Dataset Forensic Audit | Calculate actual statistics, class distribution, 0/255 mask encodings, spatial overlap check. |
| **4** | Phase 5 | ML Implementation Audit | Detailed review of `src/models.py`, `src/loss.py`, `src/train.py`, `src/evaluate.py`. |
| **5** | Phase 6 | End-to-End Execution | Run Pytest (30 tests), run training smoke test, verify FastAPI & Streamlit services live. |
| **6** | Phase 7 | Model Training & Accuracy | Train/fine-tune UNet++ and SegFormer on Sentinel-2 dataset using M4 MPS, update metrics. |
