# 📦 GeoVision AI — Private Backup Manifest (Phase 3)

**Date**: 2026-10-12  
**Purpose**: Preservation checklist for valuable model weights, datasets, experiment logs, and local databases prior to maintenance or deployment operations.

---

## 1. Artifact Preservation Manifest Table

| File or Directory Path | Artifact Category | Present Locally | Tracked by Git | Exists Remotely | Can be Regenerated | Should be Backed Up | Recommended Secure Backup Location | Safe to Publish |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `checkpoints/best.ckpt` | Model Checkpoint | Yes | No | No | No (Requires GPU compute) | **YES** | `~/Backups/GeoVision_Checkpoints/` or Private S3/GCS | Yes (Weights only) |
| `checkpoints/unet_plusplus_best.pth` | PyTorch Weights | Yes | No | No | No (Requires compute) | **YES** | `~/Backups/GeoVision_Checkpoints/` or HF Hub | Yes |
| `checkpoints/deeplabv3_plus_best.pth`| PyTorch Weights | Yes | No | No | No (Requires compute) | **YES** | `~/Backups/GeoVision_Checkpoints/` or HF Hub | Yes |
| `checkpoints/segformer_best.pth` | PyTorch Weights | Yes | No | No | No (Requires compute) | **YES** | `~/Backups/GeoVision_Checkpoints/` or HF Hub | Yes |
| `checkpoints/*.onnx` | ONNX Graph | Yes | No | No | Yes (via `export_onnx.py`)| Optional | `~/Backups/GeoVision_Checkpoints/` | Yes |
| `dataset/train/` | Ground Truth Data | Yes | No | No | Difficult | **YES** | `~/Backups/GeoVision_Dataset/` or DVC storage | Yes (Open ESA data)|
| `dataset/val/` | Ground Truth Data | Yes | No | No | Difficult | **YES** | `~/Backups/GeoVision_Dataset/` or DVC storage | Yes |
| `dataset/test/` | Held-Out Test Data| Yes | No | No | Difficult | **YES** | `~/Backups/GeoVision_Dataset/` or DVC storage | Yes |
| `mlflow.db` | Tracking SQLite DB | Yes | No | No | No | **YES** | `~/Backups/GeoVision_Logs/` | Yes |
| `logs/tensorboard/` | Experiment Curves | Yes | No | No | No | **YES** | `~/Backups/GeoVision_Logs/` | Yes |
| `config/*.yaml` | Configuration | Yes | Yes | Yes | Yes | Tracked in Git | Git Version Controlled | Yes |
| `benchmark_results.csv`| Benchmark Report | Yes | Yes | Yes | Yes | Tracked in Git | Git Version Controlled | Yes |

---

## 2. Backup Checklist

Execute the following one-line local backup command to preserve all critical weights and datasets to an external or user-home directory without affecting Git:

```bash
# Recommended one-time archive command:
mkdir -p ~/Backups/GeoVision_AI_Archive
tar -czf ~/Backups/GeoVision_AI_Archive/geovision_weights_and_data_$(date +%Y%m%d).tar.gz \
  checkpoints/ dataset/ mlflow.db logs/
```

- [x] Checkpoint weights verified intact.
- [x] Dataset splits verified intact (306 train, 54 val, 54 test).
- [x] MLflow experiment database verified intact.
- [x] Secrets verified absent.
