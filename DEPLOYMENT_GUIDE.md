# GeoVision AI — Production Deployment Guide

This guide details the deployment architectures, hardware specifications, container configurations, and cloud hosting instructions for the GeoVision AI platform.

---

## 1. Hosting Platform Evaluation

| Platform | Recommended Role | Tier Specifications | Strengths | Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Streamlit Community Cloud** | **Live Production Demo** (Verified) | 1.0 GB RAM (burst 1.5 GB), shared CPU | **100% Free** forever; zero credit card required; direct GitHub synchronization; verified 638 MB peak inference RSS. | Sleeps on dormant inactivity (~15s wake). |
| **Hugging Face Hub** | **Model Checkpoint Registry** (Verified) | Git LFS, free public model hosting | Bitwise cryptographic verification (`d21fa1...`); pinned revision locking; reliable 64 KB block streaming. | Model hosting only. |
| **Docker (Local / Hugging Face Spaces)** | **Unified Container Serving** (Verified) | Standard Docker Engine (port 7860) | Single-service container running Streamlit on port 7860 with in-process PyTorch + ONNX Runtime engines. | Requires host Docker daemon. |
| **Render / Koyeb** | *Not Recommended for ML* | 512 MB RAM | Simple Git deployment. | **Fails with OOMKilled**: PyTorch + FPN model requires ~589 MB minimum. |

### Verified Live Deployments
* **Live Web Dashboard**: [https://ai-geovision.streamlit.app/](https://ai-geovision.streamlit.app/)
* **Verified Model Registry**: [`safalya06/geovision-water-segmentation-models@441a990`](https://huggingface.co/safalya06/geovision-water-segmentation-models)
* **Authoritative Commit Pin**: `441a99034b36336c34d86c23cfe9c1a8114ca3a4`
* **Champion Model (FPN-ResNet34)**: SHA-256 `d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1`

*(Architectural Notice: The legacy FastAPI microservice has been retired from production to eliminate duplicate proxy hops, network latency, and multi-process server overhead. All inference now executes directly in-process via PyTorch and ONNX Runtime).*

---

## 2. Docker Container Specifications

### Root `Dockerfile`
The root `Dockerfile` provides a standardized, hardened production environment conforming to **Hugging Face Spaces Docker SDK** requirements:
- **Base Image**: `python:3.10-slim` (minimal attack surface, ~150 MB).
- **System Dependencies**: `libgl1`, `libglib2.0-0`, `curl` (headless OpenCV image decoding and health checks).
- **PyTorch Installation**: Optimized CPU wheels from `https://download.pytorch.org/whl/cpu` to avoid multi-gigabyte CUDA bloat.
- **Non-Root Execution**: Runs under unprivileged user `appuser` (`UID 1000`, `GID 1000`), satisfying Hugging Face Spaces security sandbox requirements.
- **Port Exposure**: Public port `7860` (Hugging Face Spaces default HTTP route for Streamlit dashboard).
- **Process Supervisor**: Managed by `docker/entrypoint.sh`, trapping termination signals (SIGTERM, SIGINT) and cleanly executing Streamlit.
- **Healthcheck**: Regular HTTP query to `http://127.0.0.1:7860/_stcore/health`.

### Build & Run Commands
```bash
# 1. Build the unified image
docker build -t geovision-ai:latest .

# 2. Run container (Streamlit UI mapped to port 7860)
docker run -d \
  --name geovision-app \
  -p 7860:7860 \
  geovision-ai:latest

# 3. Verify healthcheck status
docker inspect --format='{{json .State.Health.Status}}' geovision-app

# 4. View startup logs
docker logs -f geovision-app
```

---

## 3. Container Stack (Docker Compose)

For local development or dedicated server deployments:

```bash
docker compose up -d --build
```

Access the application at `http://localhost:7860`.

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
   The root `Dockerfile` and `docker/entrypoint.sh` handle execution automatically.
   Hugging Face Spaces routes public HTTP traffic to port **7860**, where Streamlit listens.

3. **Model Checkpoint Management on Spaces**:
   - Since model binaries (`.pth`, `.onnx`) are gitignored to maintain a lightweight repository, `src/utils/model_registry.py` provides an automated download mechanism.
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
- **Zero-Uninitialized Policy**: Inference with randomly initialized weights is strictly prohibited. If weights are missing, the UI alerts the user and halts execution.
- **Input Sanitization & Photographic Safety**: The input validator (`src/utils/input_validator.py`) inspects EXIF camera tags and color saturation, safely rejecting consumer photos (portraits, flowers, indoor objects) to prevent unscientific hydrological claims.
- **Payload Size Limits**: Strict 10 MB maximum upload size enforced to prevent denial-of-service memory exhaustion.
- **Non-Root Execution**: Container runs strictly as unprivileged user `appuser` (UID 1000).
- **Graceful Termination**: Supervisor traps SIGTERM/SIGINT signals to cleanly flush open handles and stop subprocesses without data corruption.
