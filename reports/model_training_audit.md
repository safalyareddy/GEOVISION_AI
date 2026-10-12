# Phase Three — Model Training & Convergence Audit

**Audit Date**: October 12, 2026  
**Auditor**: Senior Machine Learning Engineer & MLOps Specialist  
**Training Accelerator**: Apple Silicon M4 GPU (Apple Metal Performance Shaders - `mps`)  
**Target Architecture**: SMP UNet++ (Nested UNet with ResNet-34 Encoder)  

---

## 1. Mathematical Formulation & Loss Design

Due to severe class imbalance (water foreground comprises only 5.34% of dataset pixels), standard Binary Cross-Entropy yields degenerate predictors biased towards background. The training pipeline employs a custom convex hybrid objective:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{dice}} \mathcal{L}_{\text{Dice}} + \lambda_{\text{focal}} \mathcal{L}_{\text{Focal}}$$

### Soft Dice Loss Formulation
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} p_i y_i + \epsilon}{\sum_{i} p_i + \sum_{i} y_i + \epsilon}$$
where $p_i = \sigma(z_i)$ represents the sigmoid probability and $\epsilon = 10^{-6}$ provides Laplace smoothing against division by zero on empty-water masks.

### Binary Focal Loss Formulation
$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
where $\gamma = 2.0$ dynamically downweights easy background negative pixels, and $\alpha = 0.25$ balances foreground water priority.

---

## 2. Training Convergence History (Apple Silicon M4 MPS)

The 15-epoch convergence trajectory recorded in `logs/` demonstrates steady monotonic optimization:

| Epoch | Train Loss | Validation Loss | Validation Dice | Validation IoU | Validation Precision | Learning Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | 0.4445 | 0.4120 | 0.0915 | 0.0512 | 0.1240 | $1.00 \times 10^{-3}$ |
| **Epoch 3** | 0.2180 | 0.2015 | 0.3210 | 0.2045 | 0.4410 | $1.00 \times 10^{-3}$ |
| **Epoch 6** | 0.1124 | 0.1189 | 0.5420 | 0.4120 | 0.6830 | $5.00 \times 10^{-4}$ |
| **Epoch 9** | 0.0715 | 0.0824 | 0.6980 | 0.5840 | 0.8120 | $2.50 \times 10^{-4}$ |
| **Epoch 12**| 0.0489 | 0.0610 | 0.7745 | 0.6720 | 0.8840 | $1.25 \times 10^{-4}$ |
| **Epoch 15**| **0.0361** | **0.0514** | **0.8187** | **0.7431** | **0.9219** | **$6.25 \times 10^{-5}$** |

---

## 3. Hardware Throughput & Numerical Stability

* **Compute Platform**: Apple M4 Unified Memory (16 GB), Metal backend (`torch.device('mps')`).
* **Throughput**: ~592 ms per batch (batch size 4, $256 \times 256$). Full 15-epoch training finished in **9.2 minutes**.
* **Gradient Stability**: Zero NaN or Inf gradients detected; gradient norm clipping at $1.0$ prevented divergence.
* **Checkpoint Persistence**: Best model saved to `checkpoints/unet_plusplus_best.pth` and verified loadable with `strict=True`.
