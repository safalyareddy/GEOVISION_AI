# GeoVision AI — Forensic Baseline Audit

**Audit Date**: October 13, 2026  
**Auditor**: Senior Machine Learning & MLOps Forensic Auditor  
**Repository Target**: `https://github.com/safalyareddy/GEOVISION_AI`  
**Working Directory Root**: `/Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform`  
**Host Environment**: macOS (Darwin 24.3.0, Apple Silicon M4, 16 GB RAM)  
**Disk Space Available**: ~937 MiB on `/dev/disk3s1s1` (APFS Container)  

---

## 1. Repository Identity & Git Integrity

* **Repository Root**: `/Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform`
* **Active Branch**: `main` (tracking `origin/main`)
* **Local HEAD SHA**: `5b92e751517b7e7e272af85011be8c69d7bfbe71` (`5b92e75`)
* **Remote HEAD SHA**: `5b92e751517b7e7e272af85011be8c69d7bfbe71` (`5b92e75`)
* **Working Tree State**: Clean (`nothing to commit, working tree clean`)
* **Remotes Configured**:
  - `origin`: `https://github.com/safalyareddy/GEOVISION_AI.git` (Active primary remote)
  - `target_repo`: `https://github.com/safalyareddy/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform.git` (Legacy remote, strictly untouched per safety rules)
* **Pre-Existing Baseline Backup**: Branch `backup/phase0_baseline_09d214f` preserved at commit `09d214f`.

---

## 2. Remote CI Pipeline Verification

* **Workflow File**: `.github/workflows/ci.yml`
* **Trigger Event**: Push to `main`
* **Verified CI Run ID**: `38438912504`
* **Run URL**: [https://github.com/safalyareddy/GEOVISION_AI/actions/runs/38438912504](https://github.com/safalyareddy/GEOVISION_AI/actions/runs/38438912504)
* **Job Database ID**: `115440068193` (`test` on `ubuntu-latest`, Python 3.10)
* **Execution Timestamp**: Started `2026-10-13T17:28:38Z`, Completed `2026-10-13T17:31:32Z` (Total: 3m 7s)
* **Job Conclusion**: `success` (Green Check `✓`)
* **Test Summary**: **38 passed in 68.52s with ZERO (0) Python warnings**.

### Critical Forensic Gap: What Remote CI Does and Does Not Test
1. **What CI Tests**:
   - Syntactical execution of unit tests across `tests/test_api.py`, `tests/test_dataset.py`, `tests/test_evaluate.py`, `tests/test_models.py`, `tests/test_onnx.py`, `tests/test_train.py`.
   - Execution against CPU PyTorch wheels on Ubuntu runner.
2. **What CI Omits**:
   - **Trained Checkpoint Validation**: Weight files (`checkpoints/*.pth`, >2 GB) are excluded via `.gitignore` and not stored in Git LFS. CI runs model builder instantiation and tests either without weights or with unweighted initializations.
   - **Model Accuracy & Generalization**: No evaluation pipeline or benchmark metrics are asserted in CI.
   - **Docker Container Build & Runtime**: No Docker container image build, container startup, or health-check probe exists in `.github/workflows/ci.yml`.
   - **Dataset Split Leakage / Manifest Verification**: Dataset integrity and perceptual hash checks are not part of automated CI.
   - **Streamlit Web Dashboard**: UI container and frontend interactivity are unverified in CI.

---

## 3. Mandatory Safety & Resource Protocol

1. **Held-Out Test Set**: The 54 test tiles in `dataset/test/` (and `data/raw/test/`) are strictly frozen. No hyperparameter tuning, threshold calibration, or architecture iteration will touch the test set.
2. **Host Storage Constraints**: Available host disk space is **937 MiB**. High-throughput operations, uncompressed image dumps, or large cache writes are prohibited.
3. **Git History & Remotes**: No force pushes, no history rewrites, no pushes to `target_repo`. Pushes to `origin` require explicit user authorization.
