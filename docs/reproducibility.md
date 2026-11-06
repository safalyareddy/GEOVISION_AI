# GeoVision AI — Reproducibility & Environment Guide

This document provides deterministic, step-by-step instructions to reproduce the test environment, validate dataset integrity, evaluate trained model checkpoints, and verify PyTorch + ONNX Runtime parity.

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

Run the complete 136-test suite covering data loaders, model construction, loss functions, inference engines, model registry cryptographic verification, input validation safety gates, and ONNX parity:

```bash
PYTHONPATH=. pytest tests/ -v
```

All 136 tests must pass with 0 failures and 0 warnings.

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

To reproduce the four-model scientific benchmark across all 54 test tiles:
```bash
python src/benchmark_comparison.py --dataset_dir dataset/test --output reports/four_model_scientific_benchmark.json --device cpu
```

The verified results are tracked in [`reports/benchmark-results.csv`](../reports/benchmark-results.csv) and documented in [`reports/model-evaluation.md`](../reports/model-evaluation.md).

---

## 5. Multi-Tile 8-Combination Parity Verification (PyTorch × ONNX)

To verify the 8 model-engine combinations across 7 diverse Sentinel-2 test scenes:

```bash
python scripts/verify_8_combinations.py
```

This script:
1. Validates all 4 PyTorch architectures against their cryptographic SHA-256 digests.
2. Evaluates or exports the 4 corresponding ONNX computational graphs.
3. Tests both engines across 7 distinct test scenes (low-water, mixed, boundary demo patch, and water-rich).
4. Verifies 100.0% binary mask agreement and records results in `backend_acceptance_matrix.json`.

---

## 6. Local Serving Verification

Launch the Streamlit interactive dashboard:
```bash
streamlit run app.py --server.port 8501
```
Open `http://localhost:8501` to test interactive inference, engine toggling (PyTorch vs ONNX Runtime), threshold calibration, and multi-model comparison.

*(Architectural Notice: The legacy FastAPI microservice has been retired from production to eliminate proxy overhead and multi-process server dependencies).*

---

## 7. Docker Container Execution

```bash
# Build unified container
docker build -t geovision-ai:latest .

# Run inference service on port 7860
docker run -p 7860:7860 geovision-ai:latest
```

Access the application at `http://localhost:7860`.
