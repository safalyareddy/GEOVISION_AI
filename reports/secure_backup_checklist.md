# Secure Backup Checklist & Restoration Guide

**Document Purpose**: Provide reproducible, zero-data-loss instructions to securely archive and restore valuable machine learning artifacts (`checkpoints/`, `dataset/`, `mlflow.db`, and `logs/`) without exposing them to public Git tracking.

---

## 1. Inventory of Valued Assets Requiring Backup

| Asset Name | Local Path | Disk Size | File Count | Criticality |
| :--- | :--- | :---: | :---: | :--- |
| **Model Weights & ONNX Models** | `checkpoints/` | 2.2 GB | 17 files | **Critical** (Converged UNet++ ResNet34, ONNX runtime graphs) |
| **Satellite Imagery & Masks** | `dataset/` | 28 MB | 832 files | **Critical** (306 train, 54 val, 54 test patches + CSV splits) |
| **MLflow Experiment Database** | `mlflow.db` | 856 KB | 1 file | **Medium** (11 training runs, 228 metric rows) |
| **TensorBoard Logs** | `logs/` | 132 KB | 30 files | **Low-Medium** (Loss and validation curves) |

---

## 2. Local Cold Storage Archive Procedure

Execute the following commands from the workspace root to generate a compressed, timestamped tarball archive in a safe external directory (e.g. `~/GeoVision_Backups` or external drive):

```bash
# 1. Create a dedicated backup destination directory outside the git repository
mkdir -p ~/GeoVision_Backups

# 2. Generate a compressed tarball of all heavy artifacts
tar -czvf ~/GeoVision_Backups/geovision_artifacts_$(date +%Y%m%d_%H%M%S).tar.gz \
    checkpoints/ \
    dataset/ \
    mlflow.db \
    logs/

# 3. Generate SHA-256 integrity checksum
shasum -a 256 ~/GeoVision_Backups/geovision_artifacts_*.tar.gz > ~/GeoVision_Backups/checksums.sha256

# 4. Verify archive integrity
shasum -a 256 -c ~/GeoVision_Backups/checksums.sha256
```

---

## 3. Remote Cloud / Production Storage Alternatives

### Option A: Hugging Face Model Hub (Recommended for Checkpoints)
To store converged weights privately or publicly with Git LFS support:
```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Authenticate
huggingface-cli login

# Upload weights to your private model repo
huggingface-cli upload safalyareddy/geovision-unetplusplus checkpoints/unet_plusplus_best.pth unet_plusplus_best.pth --repo-type model
huggingface-cli upload safalyareddy/geovision-unetplusplus checkpoints/unet_plusplus_best.onnx unet_plusplus_best.onnx --repo-type model
```

### Option B: Data Version Control (DVC) with AWS S3 / Cloud Storage
```bash
# Initialize DVC
dvc init

# Track dataset and checkpoints
dvc add dataset checkpoints

# Push to remote bucket (e.g., S3 or Google Cloud Storage)
dvc remote add -d myremote s3://my-geovision-storage/dvc
dvc push
```

---

## 4. Disaster Recovery & Restoration Steps

If a clean clone of the repository is set up on a new machine or instance:

```bash
# 1. Clone repository
git clone https://github.com/safalyareddy/GEOVISION_AI.git
cd GEOVISION_AI

# 2. Unpack backup tarball
tar -xzvf ~/GeoVision_Backups/geovision_artifacts_<timestamp>.tar.gz -C .

# 3. Verify files are properly ignored by Git
git status
# Output must show: 'nothing to commit, working tree clean'

# 4. Run automated test suite to confirm weight & pipeline integrity
pytest tests/
```
