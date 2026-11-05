# GeoVision AI — Production Deployment Guide

This guide details the deployment architectures, hardware requirements, container specifications, and cloud hosting instructions for the GeoVision AI platform.

---

## 1. Hosting Platform Evaluation

| Platform | Recommended Role | Tier Specifications | Strengths | Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Streamlit Community Cloud** | **Live Production Demo** (Verified) | 1.0 GB RAM (burst 1.5 GB), shared CPU | **100% Free** forever; zero credit card required; direct GitHub synchronization; verified 638 MB peak inference RSS. | Sleeps on dormant inactivity (~15s wake). |
| **Hugging Face Hub** | **Model Checkpoint Registry** (Verified) | Git LFS, free public model hosting | Bitwise cryptographic verification (`d21fa1...`); pinned revision locking; reliable 64 KB block streaming. | Model hosting only; Docker Spaces requires PRO subscription. |
| **Docker (Local / Self-Hosted)** | **Multi-Service Microservice** | Standard Docker Engine | Multi-process supervisor (`docker/entrypoint.sh`), internal FastAPI `:8000`, public UI `:7860`. | Requires host Docker daemon. |
| **Render / Koyeb** | *Not Recommended for ML* | 512 MB RAM | Simple Git deployment. | **Fails with OOMKilled**: PyTorch + FPN model requires ~589 MB minimum. |

### Verified Live Deployments
* **Live Web Dashboard**: [https://ai-geovision.streamlit.app/](https://ai-geovision.streamlit.app/)
* **Verified Champion Checkpoint**: [`safalya06/geovision-water-segmentation-models@022039f`](https://huggingface.co/safalya06/geovision-water-segmentation-models)
* **Cryptographic Checksum**: `d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1`

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
   - If not found locally, it automatically downloads from the Hugging Face Model Hub (configured via `GEOVISION_HF_REPO`, default `safalya06/geovision-water-segmentation-models`).
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
