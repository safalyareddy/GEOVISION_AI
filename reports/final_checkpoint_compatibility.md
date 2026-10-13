# Final Checkpoint Compatibility & Weights Provenance Audit

**Audit Date**: October 13, 2026  
**Auditor**: Lead Computer Vision & Deep Learning Systems Engineer  
**Scope**: Verification of Advertised Checkpoints (`unet_plusplus_best.pth`, `deeplabv3_plus_best.pth`, `segformer_best.pth`)

---

## 1. Architecture Matching & Strict Parameter Deserialization

Every advertised model architecture was instantiated with its declared configuration and evaluated for strict deserialization (`model.load_state_dict(state, strict=True)`) using safe weights loading (`weights_only=True`):

| Model Name | Backbone / Architecture | Checkpoint File | Checkpoint Size | Parameter Count | Missing Keys | Unexpected Keys | Strict Loading | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNet++** | ResNet-34 (`smp.UnetPlusPlus`) | `checkpoints/unet_plusplus_best.pth` | 99.68 MB | 26,081,377 | **0** | **0** | **YES** | **PASS — VERIFIED** |
| **DeepLabV3+** | ResNet-34 (`smp.DeepLabV3Plus`)| `checkpoints/deeplabv3_plus_best.pth`| 85.76 MB | 22,442,753 | **0** | **0** | **YES** | **PASS — VERIFIED** |
| **SegFormer** | MiT-B2 (`HuggingFaceSegFormerWrapper`)| `checkpoints/segformer_best.pth` | 104.48 MB | 27,351,649 | **0** | **0** | **YES** | **PASS — VERIFIED** |

* **Discrepancy Resolution**: Historical reports cited non-strict fallback for DeepLabV3+ and SegFormer due to wrapper key prefix mismatches (`segformer.` namespace). In the current codebase, `HuggingFaceSegFormerWrapper` properly aligns the parameter hierarchy, enabling strict deserialization with **0 missing and 0 unexpected keys across all three architectures**.

---

## 2. Real Forward Pass & Output Sanity Verification

Each model was tested with a real Sentinel-2 satellite tile input tensor ($1 \times 3 \times 256 \times 256$, float32, scaled $[0, 1]$):

| Architecture | Output Tensor Shape | Output Dtype | Finite / NaN-Free | Sigmoid Prob Range | Mean Probability | Predicted Water Coverage on Val Tile 0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UNet++** | `[1, 1, 256, 256]` | `torch.float32` | **YES** | $[0.0005, 1.0000]$ | **0.0541** (5.41%) | **4.09%** (Selective, sharp mask) |
| **DeepLabV3+** | `[1, 1, 256, 256]` | `torch.float32` | **YES** | $[0.6734, 1.0000]$ | **0.9999** (99.99%)| **100.00%** (Degenerate all-water) |
| **SegFormer** | `[1, 1, 256, 256]` | `torch.float32` | **YES** | $[0.4681, 0.8678]$ | **0.8268** (82.68%)| **99.98%** (Degenerate all-water) |

---

## 3. Weight Provenance & Functional Findings

1. **UNet++ (Trained & Selective)**:
   - Checkpoint contains genuinely learned, discriminative spatial features.
   - On in-domain validation tile `patch_0000.png` (ground truth water prevalence $5.51\%$), it predicts $4.09\%$ coverage with high selectivity ($TN = 3.33M$ pixels across validation).
   - Provenance: 15-epoch convergence on Apple Silicon M4 MPS using Hybrid Dice-Focal Loss.
2. **DeepLabV3+ & SegFormer (Degenerate / Untrained Baseline Checkpoints)**:
   - While parameter shapes and state dict mappings are 100% strictly compatible, the checkpoint weights represent untrained or collapsed runs with extreme positive logit bias.
   - DeepLabV3+ produces output logits resulting in probabilities $> 0.67$ across all pixels, classifying 100% of the scene as water.
   - SegFormer produces output probabilities with a mean of $0.8268$, classifying $99.98\%$ of pixels as water.
   - **Conclusion**: Only UNet++ is a valid, functioning segmentation model. DeepLabV3+ and SegFormer must be explicitly flagged as non-operational baselines.
