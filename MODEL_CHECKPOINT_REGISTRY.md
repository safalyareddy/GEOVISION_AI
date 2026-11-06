# GeoVision AI — Model Checkpoint Registry & Provenance Specification

This document provides the definitive cryptographic and operational registry for all segmentation checkpoints in the GeoVision AI platform.

---

## 1. Production Model Checkpoint Registry

| Field | FPN (Champion) | UNet++ (Promoted) | DeepLabV3+ | SegFormer |
| :--- | :--- | :--- | :--- | :--- |
| **Model Identifier** | `fpn` | `unet_plusplus` | `deeplabv3_plus` | `segformer` |
| **Full Display Name** | FPN–ResNet34 (Champion Model) | UNet++–ResNet34 (Improved Model) | DeepLabV3+–ResNet34 (Improved Model) | SegFormer–MiT-B0 (Improved Model) |
| **Architecture** | Feature Pyramid Network | Nested UNet (UNet++) | DeepLabV3+ | SegFormer |
| **Encoder / Backbone** | `resnet34` | `resnet34` | `resnet34` | `nvidia/mit-b0` |
| **Framework** | PyTorch (SMP) | PyTorch (SMP) | PyTorch (SMP) | PyTorch (Transformers) |
| **Canonical Checkpoint Path**| `checkpoints/fpn_best.pth` | `checkpoints/unet_plusplus_best.pth` | `checkpoints/deeplabv3_plus_best.pth` | `checkpoints/segformer_best.pth` |
| **File Size (Bytes)** | `92,779,733` | `104,528,035` | `89,936,639` | `14,949,211` |
| **File Size (MB)** | 88.48 MB | 99.69 MB | 85.77 MB | 14.26 MB |
| **Cryptographic SHA-256** | `d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1` | `497e246ef79d87f3b2255cc119e7ab8bd0045d48caf3d4710e9f6738e56c4548` | `6c7c6dd4835d29be34b9eda03bdfca878914749a526c44cade99906cc1b1b827` | `2c762a52a5aa5cafb6aecb698bf483257fd5279bce3f3af7368850ba9354e986` |
| **Total Parameter Count** | **23,155,393** | **26,078,609** | **22,437,457** | **3,714,401** |
| **Input Shape (NCHW)** | `[N, 3, 256, 256]` | `[N, 3, 256, 256]` | `[N, 3, 256, 256]` | `[N, 3, 256, 256]` |
| **Input Normalization** | RGB float32 $\in [0.0, 1.0]$ | RGB float32 $\in [0.0, 1.0]$ | RGB float32 $\in [0.0, 1.0]$ | RGB float32 $\in [0.0, 1.0]$ |
| **Output Activation** | Sigmoid | Sigmoid | Sigmoid | Sigmoid |
| **Decision Threshold ($\tau$)** | 0.50 (Calibrated: 0.60) | 0.50 (Calibrated: 0.60) | 0.50 (Calibrated: 0.60) | 0.50 (Calibrated: 0.55) |
| **Experiment Provenance** | `EXP06_FPN_FocalTversky` | `EXP01_UNetPlusPlus_Tversky` | `EXP04_DeepLabV3Plus_FocalTversky` | `EXP05_SegFormer_FocalTversky` |
| **Best Training Epoch** | Epoch 14 | Epoch 11 | Epoch 12 | Epoch 8 |
| **Training Loss Formulation**| Focal Tversky ($\beta=0.7, \alpha=0.3, \gamma=1.333$) | Tversky ($\beta=0.7, \alpha=0.3$) | Focal Tversky ($\beta=0.7, \alpha=0.3, \gamma=1.333$) | Focal Tversky ($\beta=0.7, \alpha=0.3, \gamma=1.333$) |
| **Val Macro Dice ($\tau=0.50$)**| 0.5402 | 0.6160 | 0.5358 | 0.1558 |
| **Val Recall ($\tau=0.50$)** | 78.08% | 77.70% | 84.67% | 38.67% |
| **Val Precision ($\tau=0.50$)** | 62.28% | 70.11% | 58.71% | 11.54% |
| **Test Micro Dice ($\tau=0.50$)**| **0.6323** | **0.5319** | **0.5036** | **0.4034** |
| **Test Micro IoU ($\tau=0.50$)** | **0.4624** | **0.3623** | **0.3365** | **0.2527** |
| **Test Recall ($\tau=0.50$)** | **54.24%** | **39.75%** | **37.05%** | **32.75%** |
| **Test Precision ($\tau=0.50$)**| **75.81%** | **80.37%** | **78.57%** | **52.51%** |
| **Deployment Role** | **Production Champion** | **Promoted UNet Model** | **High-Recall Baseline** | **Edge / ViT Model** |

---

## 2. Checkpoint Provenance & Historical Baselines

To maintain scientific integrity and auditability, all historical baseline checkpoints are preserved in `checkpoints/experiments/` and are never overwritten:

### Historical Baseline Checkpoints (Archival & Reference)
1. **UNet++ Baseline (Exp1):**
   - File: `checkpoints/experiments/unet_plusplus_baseline_legacy_e9ac.pth`
   - Secondary: `checkpoints/experiments/unet_plusplus_exp1.pth`
   - SHA-256: `e9ac1bed148e6f3e8926d9008fd3fe6ad490ff8ed1dee6d79b96134ef8712c9b`
   - Test Micro Dice: `0.4285` | Test Recall: `29.76%` | Test Precision: `76.55%`
   - *Status:* Replaced in production by promoted EXP01 checkpoint (+33.6% relative recall improvement).

2. **DeepLabV3+ Baseline (Exp2):**
   - File: `checkpoints/experiments/deeplabv3_plus_exp2.pth`
   - SHA-256: `cc49b2b09932c0cfde5df6ac19c9317f11e9ecb1e03de6ccc88f4c8e23a13491`
   - Test Micro Dice: `0.3795` | Test Recall: `24.07%` | Test Precision: `89.70%`
   - *Status:* Replaced in production by EXP04 Focal Tversky checkpoint (+53.9% relative recall improvement).

3. **SegFormer Baseline (Exp3):**
   - File: `checkpoints/experiments/segformer_exp3.pth`
   - SHA-256: `dc08288d58a52bf5098d9eb4aa2978dc0063903dbf92e7eb099074624d88d42c`
   - Test Micro Dice: `0.1586` | Test Recall: `8.74%` | Test Precision: `85.84%`
   - *Status:* Replaced in production by EXP05 Focal Tversky checkpoint (+274.7% relative recall improvement).

---

## 3. Checkpoint Verification Procedures

### Autonomous Verification Command
To verify the cryptographic SHA-256 digests of all checkpoints locally, run:

```bash
./venv/bin/python -c "
import hashlib
from src.utils.model_registry import MODEL_REGISTRY, resolve_checkpoint_path

print('Validating checkpoint SHA-256 checksums:')
for name, meta in MODEL_REGISTRY.items():
    path = resolve_checkpoint_path(name, allow_download=False)
    assert path and path.exists(), f'Missing file for {name}'
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    actual = h.hexdigest()
    expected = meta['expected_sha256']
    assert actual == expected, f'Checksum mismatch on {name}'
    print(f'  ✓ {name:<15}: {actual} (MATCH)')
"
```

### Strict Deserialization Test
Verify that all four state dictionaries load strictly without skipping keys:

```bash
./venv/bin/python -c "
from src.utils.model_registry import load_verified_model, SUPPORTED_MODELS

for name in SUPPORTED_MODELS:
    model = load_verified_model(name, device='cpu')
    print(f'  ✓ {name:<15}: Strictly loaded ({sum(p.numel() for p in model.parameters()):,} parameters)')
"
```

---

## 4. Production ONNX Graph Registry & Parity Verification

In addition to native PyTorch `.pth` state dictionaries, GeoVision AI provides optimized Open Neural Network Exchange (ONNX) computational graphs for high-throughput, low-latency CPU serving.

### A. ONNX Computational Graphs

| Model Identifier | Architecture | ONNX Graph Artifact | Size (MB) | ONNX Opset | SHA-256 Digest |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `fpn` | FPN-ResNet34 (Champion) | `checkpoints/onnx/fpn.onnx` | 88.5 MB | 14 / 17 | `48e60da00e2eafc91e5c96d6276c7b6c613b95360b3f1e0f084f49b3a8e91478` |
| `unet_plusplus` | UNet++-ResNet34 | `checkpoints/onnx/unet_plusplus.onnx` | 99.7 MB | 14 / 17 | `319fcd23c48744c172b25699d1543fd6a27d61335acd8c61c0eb30038d8134d3` |
| `deeplabv3_plus` | DeepLabV3+-ResNet34 | `checkpoints/onnx/deeplabv3_plus.onnx` | 85.8 MB | 14 / 17 | `f9955856cca9b1ad1aeba5c93afd6bb5525ab36271ec6f5ab4edacef143bee3e` |
| `segformer` | SegFormer-MiT-B0 | `checkpoints/onnx/segformer.onnx` | 14.3 MB | 14 / 17 | `a6e3ea41a9b1966afc40893e11e21db66eacfbb89ba0728f4e7d537605685f56` |

*External weight sidecar tensors (`.onnx.data`) are co-located in `checkpoints/onnx/` when model sizes exceed Protobuf 2GB graph limits or multi-layer splits.*

### B. Numerical Parity Across 7 Diverse Sentinel-2 Test Scenes

Evaluated using `scripts/verify_8_combinations.py` across 7 test scenes representing diverse hydrological profiles (low-water `<1%`, mixed land/water `10–15%`, complex boundary demo patch `25%`, and water-rich `>45%`):

| Architecture | Tiles Evaluated | Binary Mask Agreement | Max Abs Δp | Mean Abs Δp | Parity Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FPN-ResNet34** (Champion) | 7 | **100.00%** | $1.69 \times 10^{-5}$ | $1.21 \times 10^{-7}$ | **PASS** |
| **UNet++-ResNet34** | 7 | **100.00%** | $3.18 \times 10^{-5}$ | $1.11 \times 10^{-7}$ | **PASS** |
| **DeepLabV3+-ResNet34** | 7 | **100.00%** | $1.39 \times 10^{-5}$ | $9.09 \times 10^{-8}$ | **PASS** |
| **SegFormer-MiT-B0** | 7 | **100.00%** | $1.73 \times 10^{-6}$ | $1.30 \times 10^{-7}$ | **PASS** |

**Aggregate Verdict**: **8/8 Model-Engine Combinations Passed** with **100.0% binary mask agreement**.

### C. Hugging Face Hub Remote Provenance
- Repository: [`safalya06/geovision-water-segmentation-models`](https://huggingface.co/safalya06/geovision-water-segmentation-models)
- Authoritative Commit Pin: `441a99034b36336c34d86c23cfe9c1a8114ca3a4`

