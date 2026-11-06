# GeoVision AI — Docker Containerization & Production Deployment Guide

This guide details the containerization architecture, build instructions, and verification steps for deploying the GeoVision AI satellite water body segmentation platform in containerized cloud environments (e.g., Hugging Face Docker Spaces, AWS ECS, GCP Cloud Run, or on-premise Docker hosts).

---

## 1. Production Architecture: Streamlit + PyTorch + ONNX Runtime

As of the Final Production Completion milestone, the active deployment topology has been streamlined to a single-service container:

- **Framework:** Streamlit Interactive Dashboard
- **Inference Engines:** Native PyTorch (In-Process CPU) & ONNX Runtime (Accelerated CPU)
- **Ports:** Web dashboard exposed strictly on port `7860` (Hugging Face Spaces & Cloud PaaS standard)
- **Security:** Non-root execution as unprivileged user `appuser` (UID 1000)
- **Status of Legacy FastAPI Microservice:** Fully retired from active production deployment. All inference, cryptographic checkpoint verification, and geospatial input validation are executed directly within the in-process model runtime.

---

## 2. Docker Image Optimization & Checkpoint Packaging

The [`.dockerignore`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/.dockerignore) file is explicitly configured to maintain an efficient image size:

### Included in Container:
- Application source code in `src/`, `config/`, `app.py`.
- Production container entrypoint: `docker/entrypoint.sh`.
- Exact four production PyTorch checkpoints (or downloaded dynamically from Hugging Face Hub commit `441a99034b36336c34d86c23cfe9c1a8114ca3a4` with strict SHA-256 validation):
  - `checkpoints/fpn_best.pth` (88.48 MB, SHA-256: `d21fa18c36be...`)
  - `checkpoints/unet_plusplus_best.pth` (99.69 MB, SHA-256: `497e246ef79d...`)
  - `checkpoints/deeplabv3_plus_best.pth` (85.77 MB, SHA-256: `6c7c6dd4835d...`)
  - `checkpoints/segformer_best.pth` (14.26 MB, SHA-256: `2c762a52a5aa...`)

### Excluded from Container:
- Raw satellite datasets (`dataset/`, ~120 MB).
- Training logs, checkpoints, and artifacts (`experiments/`, `mlruns/`, `mlflow.db`).
- Python virtual environments (`venv/`, `__pycache__/`).
- Git history and repository metadata (`.git/`).

---

## 3. Step-by-Step Build & Execution Commands

### Build and Run Production Docker Container

```bash
# 1. Build Docker image
docker build -t geovision-ai:latest -f Dockerfile .

# 2. Run container (Streamlit exposed on port 7860)
docker run -d --name geovision-app \
  -p 7860:7860 \
  geovision-ai:latest

# 3. Verify health check
curl -f http://127.0.0.1:7860/_stcore/health

# 4. Access Streamlit Web UI
open http://localhost:7860

# 5. Stop and clean up container
docker stop geovision-app && docker rm geovision-app
```

### Run via Docker Compose

```bash
# 1. Build and start service in detached mode
docker compose up -d --build

# 2. Check service status
docker compose ps

# 3. Check logs
docker compose logs -f

# 4. Access dashboard
open http://localhost:7860

# 5. Stop services
docker compose down
```

---

## 4. In-Container Verification

Test inference across both engines (PyTorch & ONNX Runtime) directly inside the running container:

```bash
docker exec -it geovision-app python -c "
from PIL import Image
from src.inference import predict

img = Image.open('assets/demo/patch_0047.png')

for model in ['fpn', 'unet_plusplus', 'deeplabv3_plus', 'segformer']:
    # Test PyTorch
    res_pt = predict(img, model_name=model, threshold=0.50, engine='pytorch')
    # Test ONNX Runtime
    res_onnx = predict(img, model_name=model, threshold=0.50, engine='onnx')
    print(f'Model: {model:<15} | PT Coverage: {res_pt[\"water_coverage_pct\"]}% ({res_pt[\"latency_ms\"]} ms) | ONNX Coverage: {res_onnx[\"water_coverage_pct\"]}% ({res_onnx[\"latency_ms\"]} ms)')
"
```

---

## 5. Resource Sizing & Memory Management

All four models together occupy **~378 MB** of parameter memory in float32 precision. When loaded on demand (lazy loading):

- **Minimum Recommended Container RAM:** `2.0 GB`
- **Optimal Container RAM:** `4.0 GB`
- **CPU Cores:** 2+ vCPUs recommended for concurrent inference.

The unified runtime dynamically manages memory, caching model instances and ONNX Runtime sessions after cryptographic verification to eliminate repeated disk reads while staying comfortably within memory limits.
