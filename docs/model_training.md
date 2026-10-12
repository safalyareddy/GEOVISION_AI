# Model Training Pipeline & Optimization

---

## 1. Training Setup & Hyperparameters

* **Framework**: PyTorch Lightning 2.0+
* **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-4}$)
* **Learning Rate Schedule**: Cosine Annealing with Warm Restarts (initial LR: $10^{-3}$, minimum LR: $10^{-6}$)
* **Batch Size**: 4 per batch
* **Precision**: Mixed Precision (16-bit / 32-bit depending on device)
* **Accelerator**: Apple Silicon Metal Performance Shaders (`mps`) / CUDA / CPU fallback

---

## 2. Loss Function

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{Dice}} + 0.5 \cdot \mathcal{L}_{\text{Focal}}$$

* **Dice Loss**: Laplacian smoothed ($\epsilon=10^{-6}$) to maximize intersection over union of water bodies.
* **Focal Loss**: Modulating factor $(1 - p_t)^{\gamma}$ with $\gamma=2.0, \alpha=0.25$ to heavily downweight easy background land pixels.

---

## 3. Training Execution Command

```bash
# Train UNet++ on local dataset using Apple Silicon MPS
python src/train.py model=unet_plusplus training.epochs=15 training.device=mps
```
All training metrics and validation checkpoints are logged automatically to TensorBoard (`logs/tensorboard/`) and MLflow (`mlflow.db`).
