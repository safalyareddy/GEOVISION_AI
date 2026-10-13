# GeoVision AI — Master Gap-Closure Baseline Report

**Baseline Audit Date**: October 13, 2026  
**Auditor**: Senior Machine Learning & DevOps Engineering Lead  
**Working Directory**: `/Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform`  
**Primary Remote**: [`https://github.com/safalyareddy/GEOVISION_AI.git`](https://github.com/safalyareddy/GEOVISION_AI.git) (`origin`)  
**Safety Backup Branch**: `backup/phase0_baseline_09d214f`  

---

## 1. Git Repository State & Remote Synchronization

* **Active Branch**: `main`
* **Local HEAD Commit**: `09d214f2efa3f9c16cd273e9fa51dc6a0371ef9d` (`09d214f`)
* **Remote `origin/main` Commit**: `09d214f2efa3f9c16cd273e9fa51dc6a0371ef9d` (`09d214f`)
* **Synchronization Status**: **100% SYNCHRONIZED** (Zero commits ahead, zero commits behind).
* **Working Tree**: Completely clean (`git status --short --branch` -> `## main...origin/main`).
* **Configured Remotes**:
  - `origin`: `https://github.com/safalyareddy/GEOVISION_AI.git` (Primary active remote).
  - `target_repo`: `https://github.com/safalyareddy/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform.git` (Legacy remote, preserved untouched per safety constraints).

### Recent Git History
```text
09d214f docs(ci): record verified GitHub Actions run 38298362711 for commit 2835ce7
2835ce7 docs(audit): complete independent verification, performance gap diagnostics, and normalized scoring
ee4fe52 docs(ci): record verified passing GitHub Actions run ID and status
c55693f feat: complete master engineering remediation, security hardening, and documentation suite
810b4ed feat(model): integrate smp unet++ weights, onnx parity verification, and complete audit reports
```

---

## 2. System Hardware, Storage & Runtimes

* **Hardware Architecture**: Apple Silicon MacBook Air M4 (Apple M4 GPU via Metal Performance Shaders `mps`, 16 GB Unified RAM).
* **Storage Capacity**:
  - Filesystem: `/dev/disk3s5` mounted on `/System/Volumes/Data`
  - Total: `228 GiB`
  - Used: `200 GiB`
  - **Available**: `5.2 GiB` (98% capacity utilized).
  - *Operational Constraint*: Bounded training and evaluation jobs are strictly enforced to prevent disk starvation.
* **Python Runtime Environments**:
  - System Python: `Python 3.14.4`
  - Project Virtual Environment (`./venv`): `Python 3.11.15`
  - GitHub Actions Runner: `Python 3.10.22` (`ubuntu-latest`)

### Key Dependency Manifest (Active `./venv`)
* `torch`: `2.14.1` (Apple Metal / MPS acceleration enabled)
* `torchvision`: `0.29.1`
* `pytorch-lightning`: `2.6.6`
* `segmentation-models-pytorch`: `0.5.0`
* `transformers`: `5.19.0`
* `fastapi`: `0.143.0`
* `uvicorn`: `0.54.0`
* `streamlit`: `1.65.0`
* `onnxruntime`: `1.30.0`
* `onnx`: `1.23.2`

---

## 3. Dataset & Checkpoint Inventory

### Dataset Status (`dataset/`)
* Total Sentinel-2 $256 \times 256$ paired image-mask samples: **414 pairs**
  - `dataset/train/`: 306 images, 306 masks (3.89% ground truth water pixels, 27 hard negatives)
  - `dataset/val/`: 54 images, 54 masks (5.51% ground truth water pixels, 7 hard negatives)
  - `dataset/test/`: 54 images, 54 masks (13.38% ground truth water pixels, 0 hard negatives) — **FROZEN HELD-OUT TEST PARTITION**
* Mask integrity: Binary $\{0, 255\}$, 1-to-1 filename alignment, zero corruptions.

### Checkpoint Status (`checkpoints/`)
* `unet_plusplus_best.pth`: 104,522,639 bytes (SHA-256: `e9ac1bed148e6f3e8926d9008fd3fe6ad490ff8ed1dee6d79b96134ef8712c9b`)
* `unet_plusplus_trained.pth`: 104,523,707 bytes
* `unet_plusplus_best.onnx` + `.onnx.data`: Exported ONNX graph and weights
* `deeplabv3_plus_best.pth`: 89,930,713 bytes
* `segformer_best.pth`: 109,550,933 bytes
* `best_model.pth`, `best.ckpt`, `segformer_best.ckpt`: Legacy checkpoints preserved intact.

---

## 4. Git Hygiene, Secrets & Artifact Audit

* **Large Tracked Files Audit**:
  - Scanned all Git-indexed blobs over 1 MB. Only one file found: `notebooks/GeoVision_AI_Segmentation.ipynb` (2.84 MB, containing research visualizations).
  - Zero checkpoints, raw datasets, virtual environments, `.tar.gz` archives, or SQLite databases are tracked by Git.
* **Secret & Credential Scan**:
  - Scanned regex heuristics for AWS keys, GitHub tokens, Hugging Face tokens, private keys, and environment variables across tracked files and commit history.
  - **Result**: Zero credentials or private tokens detected in Git. Safe template `.env.example` present and allowed.
* **Ignored Files Verification**:
  - Verified that `checkpoints/`, `dataset/`, `logs/`, `mlflow.db`, `venv/`, `.pytest_cache/`, and `.DS_Store` are properly masked by `.gitignore`.

---

## 5. Current CI Workflow & GitHub Actions Status

* **Workflow Path**: [`.github/workflows/ci.yml`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/.github/workflows/ci.yml)
* **Latest Executed CI Run**: Run ID `38298903399`
  - URL: [https://github.com/safalyareddy/GEOVISION_AI/actions/runs/38298903399](https://github.com/safalyareddy/GEOVISION_AI/actions/runs/38298903399)
  - Head SHA: `09d214f2efa3f9c16cd273e9fa51dc6a0371ef9d` (matches local HEAD)
  - Status / Conclusion: `completed` / `success`
  - Tests: **35 passed, 3 warnings in 57.60s** on `ubuntu-latest`.
