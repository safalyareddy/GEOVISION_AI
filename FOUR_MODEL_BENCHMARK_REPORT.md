# GeoVision AI — Four-Model Scientific Benchmark Report

**Evaluation Date:** 2026-11-05  
**Split Evaluated:** 54 Held-Out Test Tiles (`dataset/test`)  
**Total Pixels Evaluated:** $54 \times 256 \times 256 = \mathbf{3,538,944}$  
**Ground Truth Distribution:** Water: $473,646$ px ($13.38\%$) | Land: $3,065,298$ px ($86.62\%$)  
**Protocol:** Deterministic clean PyTorch forward pass (`model.eval()`, `torch.inference_mode()`), unstandardized RGB float32 preprocessing ($x \in [0.0, 1.0]$), fixed and validation-calibrated decision thresholds.

---

## 1. Section 1: Common Benchmark (Fixed Decision Threshold $\tau = 0.50$)

The fixed threshold benchmark provides an unbiased head-to-head comparison across all four architectures under identical operational settings.

| Model / Architecture | Pixel Accuracy | Water Precision | Water Recall | Specificity | Micro Dice (F1) | Macro Dice | Micro IoU | Macro IoU | True Positives (TP) | False Negatives (FN) | False Positives (FP) | True Negatives (TN) | Mean CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FPN–ResNet34 (Champion)** | **91.56%** | **75.81%** | **54.24%** | **97.33%** | **0.6323** | **0.4580** | **0.4624** | **0.3328** | **256,886** | **216,760** | **81,953** | **2,983,345** | **50.8 ms** |
| **UNet++–ResNet34 (EXP01)** | **90.64%** | **80.37%** | **39.75%** | **98.50%** | **0.5319** | **0.3872** | **0.3623** | **0.2718** | **188,256** | **285,390** | **45,992** | **3,019,306** | **160.1 ms** |
| **DeepLabV3+–ResNet34 (EXP04)**| **90.22%** | **78.57%** | **37.05%** | **98.44%** | **0.5036** | **0.3636** | **0.3365** | **0.2542** | **175,501** | **298,145** | **47,872** | **3,017,426** | **67.2 ms** |
| **SegFormer–MiT-B0 (EXP05)** | **87.03%** | **52.51%** | **32.75%** | **95.42%** | **0.4034** | **0.3414** | **0.2527** | **0.2275** | **155,141** | **318,505** | **140,320** | **2,924,978** | **104.4 ms** |

---

## 2. Section 2: Validation-Calibrated Benchmark (Non-Test Optimized)

Optimal operating thresholds were identified strictly on the independent 54-tile **validation set** during experiment runs. They were not tuned on the test split.

| Model / Architecture | Calibrated $\tau^*$ | Pixel Accuracy | Water Precision | Water Recall | Micro Dice | Macro Dice | Micro IoU | Macro IoU | True Positives (TP) | False Negatives (FN) | False Positives (FP) | True Negatives (TN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FPN–ResNet34 (Champion)** | **0.60** | **91.56%** | **76.45%** | **53.35%** | **0.6284** | **0.4544** | **0.4582** | **0.3295** | 252,677 | 220,969 | 77,835 | 2,987,463 |
| **UNet++–ResNet34 (EXP01)** | **0.60** | **90.60%** | **80.76%** | **39.12%** | **0.5271** | **0.3835** | **0.3578** | **0.2686** | 185,289 | 288,357 | 44,151 | 3,021,147 |
| **DeepLabV3+–ResNet34 (EXP04)**| **0.60** | **90.15%** | **79.52%** | **35.54%** | **0.4913** | **0.3542** | **0.3256** | **0.2466** | 168,339 | 305,307 | 43,347 | 3,021,951 |
| **SegFormer–MiT-B0 (EXP05)** | **0.55** | **87.35%** | **54.81%** | **31.18%** | **0.3974** | **0.3372** | **0.2480** | **0.2228** | 147,678 | 325,968 | 121,794 | 2,943,504 |

---

## 3. Per-Image Distributional Metrics ($N=54$)

| Model | Median Dice | Mean Dice (Macro) | Min Dice | Max Dice | Dice Std Dev | Median IoU | Mean IoU (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FPN–ResNet34** | **0.5083** | **0.4580** | 0.0000 | **0.8067** | 0.2620 | **0.3407** | **0.3328** |
| **UNet++–ResNet34** | **0.3601** | **0.3872** | 0.0000 | 0.8123 | 0.2657 | **0.2196** | **0.2718** |
| **DeepLabV3+–ResNet34**| **0.3398** | **0.3636** | 0.0000 | 0.7712 | 0.2599 | **0.2047** | **0.2542** |
| **SegFormer–MiT-B0** | **0.3262** | **0.3414** | 0.0000 | 0.7289 | 0.2185 | **0.1949** | **0.2275** |

---

## 4. Scientific Analysis & Architectural Takeaways

### Why FPN is the Definitive Platform Champion
1. **Multi-Scale Feature Pyramid Fusion:** Optical satellite water bodies span an extreme range of spatial scales—from massive open reservoirs to narrow drainage canals and sinuous rivers only 1–3 pixels wide (10–30 meters GSD). Traditional downsampling decoders in UNet++ and DeepLabV3+ lose these thin linear structures during deep spatial downsampling. FPN retains lateral connections from high-resolution shallow layers ($P_2, P_3$), allowing it to detect **256,886 true water pixels**—outperforming UNet++ by $+68,630$ pixels and DeepLabV3+ by $+81,385$ pixels.
2. **Superior Recall (54.24%):** FPN reduces false negatives to $216,760$ (compared to $285,390$ in UNet++ and $298,145$ in DeepLabV3+), representing the lowest missed-water rate across all models.
3. **Execution Efficiency:** At **50.8 ms** per $256 \times 256$ tile on CPU, FPN runs **$3.15\times$ faster** than UNet++ (160.1 ms) due to lightweight $1 \times 1$ conv lateral projections and uniform 256-channel feature maps.

### Why UNet++ (EXP01) is the Highest Precision Model
1. **Dense Skip Aggregation:** UNet++ nested convolutional blocks filter false background activations, achieving the highest precision (**80.37%** at $\tau=0.50$ and **80.76%** at $\tau=0.60$) with only $45,992$ false positives.
2. **Balanced Contour Delineation:** Where clear reservoir boundaries exist, UNet++ produces smooth, highly confident contours with minimal coastal noise.

### DeepLabV3+ (EXP04) Characteristics
- Dilated convolutions in the Atrous Spatial Pyramid Pooling (ASPP) module allow DeepLabV3+ to capture wide contextual fields, delivering **78.57% precision** and **50.36% Micro Dice**.
- However, the absence of high-resolution dense lateral skip connections limits its ability to delineate small ponds under 5 pixels in area.

### SegFormer (EXP05) Profile
- With only **3.71 million parameters** and a **14.26 MB footprint**, SegFormer is the most compact model on the platform (85.7% fewer parameters than UNet++).
- While self-attention across large image patches provides good global land-water separation, its $4\times$ downsampled decoder logits interpolated via bilinear upsampling exhibit higher boundary blurring on satellite tiles, producing $140,320$ false positives. It remains highly suitable as an edge deployment model for constrained IoT hardware.

---

## 5. Single Command Reproducibility

To re-run this exact 54-tile benchmark and generate the accompanying JSON and CSV reports:

```bash
./venv/bin/python src/benchmark_comparison.py --dataset_dir dataset/test --output reports/four_model_scientific_benchmark.json --device cpu
```
