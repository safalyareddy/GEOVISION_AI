# GeoVision AI — Reproducibility & Environment Guide

This document provides deterministic, step-by-step instructions to reproduce the test environment, validate dataset integrity, evaluate trained model checkpoints, and verify ONNX Runtime parity.

---

## 1. Environment Setup

### Native Virtual Environment (Python 3.10 / 3.11)
```bash
# Clone repository
git clone https://github.com/safalyareddy/GEOVISION_AI.git
cd GEOVISION_AI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install CPU PyTorch wheels (or standard torch on CUDA/MPS)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 2. Automated Test Suite Validation

Run the complete 54-test suite across data loaders, model construction, loss functions, API endpoints, model registry cryptographic verification, evaluation metrics, and ONNX parity:

```bash
PYTHONPATH=. pytest tests/ -v
```

All 54 tests must pass without warnings or regressions.

---

## 3. Dataset Integrity & Pair Validation

Validate 1-to-1 matching between Sentinel-2 optical image tiles and binary water masks:

```bash
python scripts/validate_dataset.py
```

This verifies:
- Total split counts: **306 train**, **54 validation**, **54 held-out test** samples.
- Complete 1-to-1 pairing (0 unmatched pairs).
- Zero corrupted image bytes across PNG/JPEG/TIFF files.
- Spatial dimension consistency ($256 \times 256 \times 3$ RGB images, $256 \times 256 \times 1$ masks).

---

## 4. Benchmark & Metrics Reproduction

To independently evaluate saved model checkpoints on the held-out test dataset:

```bash
# Evaluate baseline models and generate benchmark report
python src/evaluate.py --dataset-dir dataset --save-csv
```

To reproduce the 2026 improved model evaluations:
```bash
# Run multi-model evaluation and visual diagnostic generation
python scripts/generate_diagnostic_visualizations.py
```

The verified results are tracked in [`reports/benchmark-results.csv`](../reports/benchmark-results.csv) and documented in [`reports/model-evaluation.md`](../reports/model-evaluation.md).

---

## 5. ONNX Export & Parity Verification

GeoVision AI supports zero-copy ONNX export for low-latency CPU production serving:

```bash
# Export PyTorch weights to ONNX graph (Opset 17)
python scripts/export_onnx.py \
    --model unet_plusplus \
    --checkpoint checkpoints/unet_plusplus_best.pth \
    --output checkpoints/unet_plusplus_best.onnx

# Verify bitwise and numerical parity between PyTorch and ONNX Runtime (< 1e-4 tolerance)
python scripts/debug_inference.py \
    --image dataset/test/images/patch_0047.png \
    --checkpoint checkpoints/unet_plusplus_best.pth
```

---

## 6. Local Serving Verification

### FastAPI Microservice
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Test health endpoint:
```bash
curl http://localhost:8000/health
```

### Streamlit Dashboard
```bash
streamlit run app.py --server.port 8501
```
Open `http://localhost:8501` to test interactive inference.

---

## 7. Docker Container Execution

```bash
# Build unified container
docker build -t geovision-ai:latest .

# Run inference service
docker run -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints:ro geovision-ai:latest
```
