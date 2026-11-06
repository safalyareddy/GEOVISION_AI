# Docker & Production Deployment Guide

This document details local development, containerization, and cloud deployment procedures for the GeoVision AI platform.

---

## 1. Local Development Stack (Native Python)

```bash
# 1. Clone repository
git clone https://github.com/safalyareddy/GEOVISION_AI.git
cd GEOVISION_AI

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install CPU PyTorch wheels and dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Launch Streamlit interactive dashboard (port 8501 or 7860)
streamlit run app.py --server.port 8501
```

Access the dashboard at `http://localhost:8501`.

*(Architectural Notice: The legacy FastAPI microservice has been completely retired from the active runtime architecture. All inference, checkpoint verification, and input validation are executed in-process via `src.inference`).*

---

## 2. Containerized Deployment (Docker)

The production Docker container runs a hardened, non-root Streamlit instance on port `7860`, conforming to the **Hugging Face Docker Space SDK**:

```bash
# 1. Build production image
docker build -t geovision-ai:latest .

# 2. Run container
docker run -d \
  --name geovision-app \
  -p 7860:7860 \
  geovision-ai:latest

# 3. Check health status
docker inspect --format='{{json .State.Health.Status}}' geovision-app

# 4. View logs
docker logs -f geovision-app
```

Access the application at `http://localhost:7860`.

---

## 3. Container Stack (Docker Compose)

For local development with volume mounts:

```bash
# Build and start service in background
docker compose up -d --build

# View container logs
docker compose logs -f

# Access dashboard
open http://localhost:7860
```

### Volume Mounting Strategy
Optionally mount local checkpoints read-only from the host:
```yaml
volumes:
  - ./checkpoints:/app/checkpoints:ro
```
This isolates the checkpoint binaries while protecting host files from modifications during runtime. If local checkpoints are not mounted, the application automatically downloads cryptographically verified weights from Hugging Face Hub (`safalya06/geovision-water-segmentation-models@441a990`).

---

## 4. Live Cloud Platforms

- **Live Streamlit Cloud Deployment**: [https://ai-geovision.streamlit.app/](https://ai-geovision.streamlit.app/)
- **Hugging Face Model Hub**: [https://huggingface.co/safalya06/geovision-water-segmentation-models](https://huggingface.co/safalya06/geovision-water-segmentation-models)
- **Authoritative Commit Pin**: `441a99034b36336c34d86c23cfe9c1a8114ca3a4`
