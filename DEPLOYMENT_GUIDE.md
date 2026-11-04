# GeoVision AI — Production Deployment Guide

This guide details the deployment architectures, hardware requirements, container specifications, and cloud hosting instructions for the GeoVision AI platform.

---

## 1. Hosting Platform Evaluation

| Platform | Recommended Role | Free Tier Specifications | Strengths | Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Hugging Face Spaces** | **Primary Interactive Demo** (Docker SDK) | 2 vCPU, 16 GB RAM, 50 GB Disk | Zero hosting cost; 16 GB RAM accommodates PyTorch model weights effortlessly; direct ML community reach. | Cold start on dormant free spaces (~30s). |
| **Railway** | **Inference API Microservice** (FastAPI) | $5 credit / month (pay-as-you-go) | Native Docker support; automatic SSL/TLS; isolated environment variables; low latency. | Paid usage beyond free tier. |
| **Render** | Alternative API Service | 0.5 CPU, 512 MB RAM | Simple Git deployment. | 512 MB free RAM is tight for multiple simultaneous PyTorch models. |
| **Vercel / Netlify** | *Not Recommended* | Serverless Lambdas | Fast static CDN edge. | 50 MB / 250 MB function size limits cannot accommodate PyTorch dependencies. |

### Architectural Recommendation
1. **Public Interactive ML Portfolio Demo**: Deploy to **Hugging Face Spaces** using the **Docker SDK**. The generous 16 GB RAM allowance ensures seamless model inference and multi-architecture switching without Out-Of-Memory (OOM) risks.
2. **Production API Endpoint**: Deploy the FastAPI microservice to **Railway** or a cloud container instance using `Dockerfile`.

---

## 2. Docker Container Specifications

### Root `Dockerfile`
The root `Dockerfile` provides a standardized, hardened production environment conforming to Hugging Face Spaces Docker SDK requirements:
- **Base Image**: `python:3.10-slim` (minimal attack surface, ~150 MB).
- **System Dependencies**: `libgl1`, `libglib2.0-0`, `curl` (headless OpenCV image decoding and health checks).
- **PyTorch Installation**: Optimized CPU wheels from `https://download.pytorch.org/whl/cpu` to avoid multi-gigabyte CUDA bloat.
- **Non-Root Execution**: Runs under unprivileged user `appuser` (`UID 1000`, `GID 1000`), satisfying Hugging Face Spaces security sandbox requirements.
- **Port Exposure**: Public port `7860` (Hugging Face Spaces default HTTP route for Streamlit dashboard) and internal port `8000` (FastAPI).
- **Process Supervision**: Managed by `docker/entrypoint.sh`, trapping termination signals (SIGTERM, SIGINT) and gracefully shutting down both Uvicorn and Streamlit.
- **Healthcheck**: Regular HTTP query to internal microservice `http://127.0.0.1:8000/health`.

### Build & Run Commands
```bash
# 1. Build the unified image
docker build -t geovision-ai:latest .

# 2. Run container (Streamlit UI mapped to port 7860)
docker run -d \
  --name geovision-space \
  -p 7860:7860 \
  geovision-ai:latest

# 3. Verify healthcheck status
docker inspect --format='{{json .State.Health.Status}}' geovision-space

# 4. View supervisor startup logs
docker logs -f geovision-space
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

## 4. Deploying to Hugging Face Spaces (Docker SDK)

1. **Create Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/new-space).
   - Space Name: `geovision-ai`
   - License: `MIT`
   - SDK: **Docker** (Blank)
   - Hardware: **CPU Basic** (2 vCPU, 16 GB RAM — Free)
   - Visibility: **Public**

2. **Repository Configuration**:
   The root `Dockerfile` and `docker/entrypoint.sh` handle process execution automatically.
   Hugging Face Spaces routes HTTP traffic to port **7860**, where Streamlit listens. Streamlit communicates with FastAPI internally via `http://127.0.0.1:8000`.

3. **Model Checkpoint Management on Spaces**:
   - Since large model binaries (`.pth`) are gitignored to maintain a lightweight repository, `src/utils/model_registry.py` provides an automated download mechanism.
   - When the container boots, `docker/entrypoint.sh` checks for the champion FPN model (`fpn_best.pth`).
   - If not found locally, it automatically downloads from the Hugging Face Model Hub (configured via `GEOVISION_HF_REPO`, default `safalyareddy/geovision-water-segmentation-models`).
   - Cryptographic SHA-256 validation (`d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1`) ensures bitwise model authenticity.

4. **Git Remote Setup & Deployment**:
   ```bash
   # Add Hugging Face Space remote
   git remote add space https://huggingface.co/spaces/<your-username>/geovision-ai

   # Push main branch to deploy
   git push space main
   ```

---

## 5. Security & Runtime Hardening

- **Cryptographic Verification**: Every model checkpoint is validated against its expected SHA-256 hash before deserialization. Corrupted or unverified weights are rejected.
- **Zero-Uninitialized Policy**: Inference with randomly initialized weights is strictly prohibited. If weights are missing, the API returns HTTP 503 and the UI alerts the user.
- **Payload Size Limits**: Strict 10 MB maximum request body enforced in FastAPI to prevent denial-of-service memory exhaustion.
- **Input Sanitization**: File magic bytes and image headers verified with PIL `image.verify()` prior to array decoding.
- **Non-Root Execution**: Container runs strictly as unprivileged user `appuser` (UID 1000).
- **Graceful Termination**: Supervisor traps SIGTERM/SIGINT signals to cleanly flush open handles and stop subprocesses without data corruption.
