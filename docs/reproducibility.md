# Reproducibility & Environment Setup

---

## 1. Quick Start & Environment Recreation

To reproduce the verified benchmarks and environment:

```bash
# Clone the verified repository
git clone https://github.com/safalyareddy/GEOVISION_AI.git
cd GEOVISION_AI

# Create Python 3.11 environment
python3.11 -m venv venv
source venv/bin/activate

# Install verified dependencies
pip install -r requirements.txt

# Run complete automated verification test suite
pytest tests/ -v
```

---

## 2. Dataset Verification

```bash
# Verify dataset split integrity, image pairing, and corruption status
python scripts/validate_dataset.py
```
Outputs validation status and verifies MD5 hashes against `reports/dataset_split_manifest.json`.

---

## 3. ONNX Export & Parity Verification

```bash
# Export trained PyTorch model to ONNX Runtime graph
python scripts/export_onnx.py --model unet_plusplus --checkpoint checkpoints/unet_plusplus_best.pth

# Verify numerical parity (< 1e-4 tolerance)
python scripts/debug_inference.py --image dataset/test/images/patch_0047.png --checkpoint checkpoints/unet_plusplus_best.pth
```
