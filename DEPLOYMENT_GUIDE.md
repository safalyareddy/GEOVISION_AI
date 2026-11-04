# GeoVision AI — Production Deployment Guide

This guide details the deployment architectures, hardware requirements, container specifications, and cloud hosting recommendations for the GeoVision AI platform.

---

## 1. Hosting Platform Evaluation

| Platform | Recommended Role | Free Tier Specifications | Strengths | Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Hugging Face Spaces** | **Primary Interactive Demo** (Streamlit / Docker) | 2 vCPU, 16 GB RAM, 50 GB Disk | Zero hosting cost; 16 GB RAM accommodates PyTorch model weights effortlessly; direct ML community reach. | Cold start on dormant free spaces (~30s). |
| **Railway** | **Inference API Microservice** (FastAPI) | $5 credit / month (pay-as-you-go) | Native Docker support; automatic SSL/TLS; isolated environment variables; low latency. | Paid usage beyond free tier. |
| **Render** | Alternative API Service | 0.5 CPU, 512 MB RAM | Simple Git deployment. | 512 MB free RAM is tight for multiple simultaneous PyTorch models. |
| **Vercel / Netlify** | *Not Recommended* | Serverless Lambdas | Fast static CDN edge. | 50 MB / 250 MB function size limits cannot accommodate PyTorch dependencies. |

### Architectural Recommendation
1. **Public Interactive ML Portfolio Demo**: Deploy to **Hugging Face Spaces** using the Streamlit or Docker runtime. The generous 16 GB RAM allowance ensures seamless model inference and multi-architecture switching without Out-Of-Memory (OOM) risks.
2. **Production API Endpoint**: Deploy the FastAPI microservice to **Railway** or a cloud container instance using `Dockerfile`.

---

## 2. Docker Container Specifications

### Root `Dockerfile`
The root `Dockerfile` provides a standardized, secure production environment:
- **Base Image**: `python:3.10-slim` (minimal attack surface, ~150 MB).
- **System Dependencies**: `libgl1`, `libglib2.0-0` (required for headless OpenCV image decoding).
- **PyTorch Installation**: Optimized CPU wheels from `https://download.pytorch.org/whl/cpu` to avoid multi-gigabyte CUDA bloat.
- **Port Exposure**: Port `8000` (FastAPI) and Port `8501` (Streamlit).
- **Healthcheck**: Regular HTTP query to `http://localhost:8000/health`.

### Build & Run Commands
```bash
# Build the unified image
docker build -t geovision-ai:latest .

# Run FastAPI backend
docker run -d \
  --name geovision-api \
  -p 8000:8000 \
  -e PORT=8000 \
  geovision-ai:latest

# Verify healthcheck status
docker inspect --format='{{json .State.Health.Status}}' geovision-api
```

---

## 3. Multi-Container Stack (Docker Compose)

For local development or dedicated server deployments, run the decoupled microservice and dashboard:

```bash
docker compose up -d --build
```

Services configured in `docker-compose.yml`:
1. **`api`**: Exposes port `8000` for REST requests.
2. **`ui`**: Exposes port `8501` and connects to the `api` service over the internal Docker network.

---

## 4. Deploying to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces):
   - Name: `geovision-ai`
   - SDK: **Docker** or **Streamlit**
   - Visibility: **Public**
2. In the repository settings or `README.md` metadata:
```yaml
---
title: GeoVision AI Water Segmentation
emoji: 🌊
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---
```
3. Push the repository files to the Hugging Face Space git remote.

---

## 5. Security & Runtime Hardening

- **Payload Size Limits**: Strict 10 MB maximum request body enforced in FastAPI to prevent denial-of-service memory exhaustion.
- **Input Sanitization**: File magic bytes and image headers verified with PIL `image.verify()` prior to array decoding.
- **Non-Root Execution**: In production container clusters, run as an unprivileged user (`USER appuser`).
- **No Secret Exposure**: All configuration sourced from environment variables; zero hardcoded credentials or paths.
