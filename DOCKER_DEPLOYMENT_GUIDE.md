# GeoVision AI — Docker Containerization & Production Deployment Guide

This guide details the containerization architecture, build instructions, and verification steps for deploying the GeoVision AI satellite water body segmentation platform in cloud and on-premise environments.

---

## 1. Deployment Architecture Options

GeoVision AI supports two production container deployment topologies:

### Topology A: Single-Container Unified Mode (Hugging Face Spaces & Cloud PaaS)
- **File:** [`Dockerfile`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/Dockerfile)
- **Entrypoint:** [`docker/entrypoint.sh`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/docker/entrypoint.sh)
- **Ports:** Public UI on `7860` (Hugging Face Spaces standard); Internal FastAPI on `8000`.
- **Process Supervisor:** `entrypoint.sh` starts FastAPI in the background, waits for healthcheck, and launches Streamlit.
- **Security:** Runs as unprivileged `appuser` (UID 1000).

### Topology B: Multi-Container Microservice Mode (Docker Compose / Kubernetes)
- **File:** [`docker-compose.yml`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/docker-compose.yml)
- **Services:**
  1. `api`: Dedicated FastAPI microservice ([`docker/Dockerfile.api`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/docker/Dockerfile.api)) on port `8000`.
  2. `ui`: Dedicated Streamlit frontend ([`docker/Dockerfile.ui`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/docker/Dockerfile.ui)) on port `8501`.

---

## 2. Docker Image Optimization & Checkpoint Packaging

The [`.dockerignore`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/.dockerignore) file is explicitly configured to maintain an efficient image size:

### Included in Container:
- All source code in `src/`, `config/`, `app.py`.
- Exact four production PyTorch checkpoints:
  - `checkpoints/fpn_best.pth` (88.48 MB)
  - `checkpoints/unet_plusplus_best.pth` (99.69 MB)
  - `checkpoints/deeplabv3_plus_best.pth` (85.77 MB)
  - `checkpoints/segformer_best.pth` (14.26 MB)

### Excluded from Container:
- Raw satellite datasets (`dataset/`, ~120 MB).
- Training logs, checkpoints, and artifacts (`experiments/`, `mlruns/`, `mlflow.db`).
- Python virtual environments (`venv/`, `__pycache__/`).
- Git history and repository metadata (`.git/`).

---

## 3. Step-by-Step Build & Execution Commands

### Mode A: Single-Container Unified Deployment

```bash
# 1. Build Docker image
docker build -t geovision-ai:latest -f Dockerfile .

# 2. Run container
docker run -d --name geovision-app \
  -p 7860:7860 \
  -p 8000:8000 \
  geovision-ai:latest

# 3. Verify internal health check
curl -f http://127.0.0.1:8000/health

# 4. Access Streamlit Web UI
open http://localhost:7860

# 5. Stop and clean up container
docker stop geovision-app && docker rm geovision-app
```

### Mode B: Docker Compose Multi-Service Deployment

```bash
# 1. Build and start services in detached mode
docker compose up -d --build

# 2. Check service status
docker compose ps

# 3. Check logs
docker compose logs -f

# 4. Access services
#    • FastAPI Docs:  http://localhost:8000/docs
#    • Streamlit UI:  http://localhost:8501

# 5. Stop services
docker compose down
```

---

## 4. Live API Verification Through Docker

Test inference across all four models running inside the Docker container:

```bash
# Test healthcheck
curl -s http://127.0.0.1:8000/health

# Query available models
curl -s http://127.0.0.1:8000/models

# Run inference test using Python
python3 -c "
import httpx

with open('dataset/test/images/patch_0000.png', 'rb') as f:
    img_bytes = f.read()

for m in ['fpn', 'unet_plusplus', 'deeplabv3_plus', 'segformer']:
    files = {'file': ('test.png', img_bytes, 'image/png')}
    data = {'model_name': m, 'threshold': 0.50}
    r = httpx.post('http://127.0.0.1:8000/predict', files=files, data=data, timeout=30.0)
    assert r.status_code == 200, f'Error: {r.text}'
    res = r.json()
    print(f'Model: {m:<15} | Coverage: {res[\"water_coverage_percentage\"]}% | Latency: {res[\"latency_ms\"]} ms')
"
```

---

## 5. Memory Management & Cloud Sizing

All four models together occupy **~378 MB** of parameter memory in float32 precision. When loaded on demand (lazy loading):

- **Minimum Recommended Container RAM:** `2.0 GB`
- **Optimal Container RAM:** `4.0 GB`
- **CPU Cores:** 2+ vCPUs recommended for concurrent inference.

The API dynamically manages memory, caching model instances after verification to eliminate repeated disk reads while staying comfortably within memory limits.
