# Docker & Production Deployment Guide

---

## 1. Local Development Stack (Native Python)

```bash
# 1. Clone repository
git clone https://github.com/safalyareddy/GEOVISION_AI.git
cd GEOVISION_AI

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Launch FastAPI backend (port 8000)
uvicorn src.api.main:app --port 8000 &

# 4. Launch Streamlit UI (port 8501)
streamlit run app.py --server.port 8501
```

---

## 2. Containerized Deployment (Docker Compose)

The multi-container stack runs the FastAPI microservice and Streamlit frontend in decoupled network services with read-only checkpoint volume mounts:

```bash
# Build and start services in background
docker compose up -d --build

# View container logs
docker compose logs -f

# Verify API health
curl http://localhost:8000/health

# Access dashboard
open http://localhost:8501
```

### Volume Mounting Strategy
Model weights are mounted read-only from the host:
```yaml
volumes:
  - ./checkpoints:/app/checkpoints:ro
```
This isolates the 2.2 GB checkpoint binaries from the container build layers while protecting host files from modifications during runtime.
