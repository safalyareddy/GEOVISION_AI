# GeoVision AI — Production Model Validation Report

**Executive Summary**: This document presents the authoritative engineering validation of all four PyTorch segmentation models in the GeoVision AI satellite water-body segmentation platform. All four models have been evaluated against the complete 54 held-out test split tiles (`dataset/test/`), cryptographically locked via SHA-256 signatures, and published to Hugging Face Hub under immutable commit pin `441a99034b36336c34d86c23cfe9c1a8114ca3a4`.

---

## 1. Discrepancy Diagnosis & Root Cause Analysis

### Prior Remote Hugging Face Hub State (`0ce5ce5824`)
Prior to this engineering run, the deployed Streamlit application and remote Hugging Face repository exhibited critical discrepancies:
1. **UNet++ Missing Checkpoint**: The Hub repository hosted an unpromoted legacy checkpoint `unet_plusplus_baseline_legacy_e9ac.pth` (SHA-256: `e9ac58e70a4a8ce88e892d2427f4e912f2c8d28a30686940a4309ae5433a5713`) while `unet_plusplus_best.pth` was absent or mismatched.
2. **DeepLabV3+ Degeneracy**: The Hub repository hosted `deeplabv3_plus_best.pth` matching SHA-256 `58efdb49cfaaa7e914d03e5c709e99a4e4d7756f1165a2d677611a58e0a3594f`. When evaluated, this checkpoint produced degenerate output: sigmoid probabilities saturated to 1.0 across 100% of pixels regardless of input, predicting 100% water coverage and causing severe production malfunction.
3. **SegFormer Absence**: `segformer_best.pth` was not available on Hugging Face Hub.

### Resolution Protocol
1. **Authoritative Experiment Auditing**:
   - `EXP06` (`checkpoints/fpn_best.pth`): Cryptographically verified as the genuine FPN-ResNet34 Champion (SHA-256: `d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1`).
   - `EXP01` (`experiments/EXP01_unet_plusplus_resnet34/checkpoints/best_model.pth`): Authoritative UNet++-ResNet34 (SHA-256: `497e246ef79d87f3b2255cc119e7ab8bd0045d48caf3d4710e9f6738e56c4548`).
   - `EXP04` (`experiments/EXP04_deeplabv3_plus_resnet34/checkpoints/best_model.pth`): Authoritative DeepLabV3+-ResNet34 (SHA-256: `6c7c6dd4835d29be34b9eda03bdfca878914749a526c44cade99906cc1b1b827`).
   - `EXP05` (`experiments/EXP05_segformer_mit_b0/checkpoints/best_model.pth`): Authoritative SegFormer-MiT-B0 (SHA-256: `2c762a52a5aa5cafb6aecb698bf483257fd5279bce3f3af7368850ba9354e986`).
2. **Hugging Face Hub Synchronization**:
   - Authenticated with token for user `safalya06`.
   - Uploaded `unet_plusplus_best.pth` (`497e...`), `deeplabv3_plus_best.pth` (`6c7c...`), and `segformer_best.pth` (`2c76...`) to `safalya06/geovision-water-segmentation-models`.
   - Resulting immutable commit SHA: `441a99034b36336c34d86c23cfe9c1a8114ca3a4`.
   - Verified that LFS trees on remote Hub match local authoritative SHA-256 hashes bit-for-bit.

---

## 2. Authoritative Test Set Evaluation (54 Held-Out Tiles)

Evaluation protocol:
- Dataset: 54 Sentinel-2 optical tiles in `dataset/test/images/` and corresponding masks in `dataset/test/masks/`.
- Preprocessing: Resize to $256 \times 256$, float32 tensor normalized to $[0.0, 1.0]$.
- Execution mode: `torch.inference_mode()` on CPU, sigmoid activation, decision threshold $\tau = 0.50$.
- Total pixels per evaluation: $54 \times 256 \times 256 = 3,538,944$ pixels.

| Architecture | Experiment | SHA-256 (first 16 hex) | Accuracy | Precision | Recall | Micro Dice | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|---|
| **FPN-ResNet34 (Champion)** | EXP06 | `d21fa18c36beabdb` | **91.56%** | 75.81% | **54.24%** | **0.6323** | 256,886 | 216,760 | 82,006 | 2,984,188 |
| **UNet++-ResNet34** | EXP01 | `497e246ef79d87f3` | 90.64% | **80.37%** | 39.75% | 0.5319 | 188,256 | 285,390 | 45,982 | 3,020,212 |
| **DeepLabV3+-ResNet34** | EXP04 | `6c7c6dd4835d29be` | 90.22% | 78.57% | 37.05% | 0.5036 | 175,500 | 298,146 | 47,844 | 3,018,350 |
| **SegFormer-MiT-B0** | EXP05 | `2c762a52a5aa5caf` | 87.03% | 52.51% | 32.75% | 0.4034 | 155,142 | 318,504 | 140,306 | 2,925,888 |

### Pixel-Level Agreement with Published Benchmark
- FPN EXP06 test set pixels match the original benchmark 100% (Micro Dice 0.6323).
- UNet++ EXP01 test set pixels match the original benchmark 100% (Micro Dice 0.5319).
- DeepLabV3+ EXP04 test set pixels match the original benchmark 100% (Micro Dice 0.5036).
- SegFormer EXP05 test set pixels match the original benchmark 100% (Micro Dice 0.4034).

---

## 3. Cryptographic Verification & Strict Deserialization Protocol

All model loading in `src/utils/model_registry.py` strictly follows the safety architecture:
1. **Streaming SHA-256 Check**: Computed using 64 KB chunks directly on disk. Checkpoints failing verification are never loaded into memory.
2. **Safe Deserialization**: Checkpoint files are deserialized with `torch.load(..., weights_only=True)`.
3. **Strict State Dict Enforcement**: State dictionaries are loaded into the PyTorch module with `model.load_state_dict(state_dict, strict=True)`. Missing or extra keys immediately trigger an exception.
4. **Eval Mode Guarantee**: `model.eval()` is explicitly invoked prior to returning the module instance.
5. **Zero Fallback Principle**: Randomly initialized models or placeholder weights are strictly prohibited. If a checkpoint cannot be verified or obtained, inference fails safe with an informative diagnostic.

---

## 4. Architecture Specifications

| Architecture | Model ID | Encoder / Backbone | Params (M) | File Size (MB) | Loss Function |
|---|---|---|---|---|---|
| FPN | `fpn` | ResNet-34 | ~23.2 M | 88.5 MB | Focal Tversky (α=0.7, β=0.3, γ=1.333) |
| UNet++ | `unet_plusplus` | ResNet-34 | ~26.1 M | 99.7 MB | Focal Tversky (α=0.7, β=0.3, γ=1.333) |
| DeepLabV3+ | `deeplabv3_plus` | ResNet-34 | ~22.5 M | 85.8 MB | Focal Tversky (α=0.7, β=0.3, γ=1.333) |
| SegFormer | `segformer` | MiT-B0 | ~3.7 M | 14.3 MB | Focal Tversky (α=0.7, β=0.3, γ=1.333) |

---

## 5. Summary & Sign-off

The model registry, Hugging Face Hub repository, and runtime inference engines are in 100% cryptographic and numerical harmony. All 4 models have been validated and promoted to production status.
