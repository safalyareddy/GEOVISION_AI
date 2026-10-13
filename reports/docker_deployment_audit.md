# Phase Nine — Docker & Deployment Configuration Audit

**Audit Date**: October 12, 2026  
**Auditor**: Production Deployment & SRE Specialist  
**Configuration Files**: [`docker-compose.yml`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/docker-compose.yml), [`docker/Dockerfile.api`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/docker/Dockerfile.api), [`docker/Dockerfile.ui`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/docker/Dockerfile.ui), [`.dockerignore`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/.dockerignore)  

---

## 1. Container Configuration Hardening

### A. Context Bloat Mitigation via `.dockerignore`
Prior to remediation, no `.dockerignore` file existed. Building the API or UI container images would copy the entire 2.2 GB `checkpoints/` directory, 28 MB `dataset/`, 39 MB `venv/`, and `.git/` history into the Docker build context.
* **Remediation**: Created `.dockerignore` strictly omitting local binaries, weights, datasets, virtual environments, and caches. Build context reduced by over **98%**.

### B. UI Entrypoint Correction in `docker/Dockerfile.ui`
* **Defect**: Legacy Dockerfile configured `CMD ["streamlit", "run", "client/app.py", ...]`. However, `client/app.py` was relocated to root `app.py`.
* **Fix**: Updated command to `CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]`.

### C. Volume Mount Path Mismatch in `docker-compose.yml`
* **Defect**: Service `api` mounted `./weights:/app/weights`. However, the API code searches `checkpoints/` relative to working directory `/app` (`/app/checkpoints`).
* **Fix**: Updated volume mount to:
  ```yaml
  volumes:
    - ./checkpoints:/app/checkpoints:ro
  ```
  The `:ro` read-only attribute prevents the running container from modifying or damaging host model checkpoints.

---

## 2. Production Deployment Architecture

```mermaid
graph LR
    Client[Web Browser] -->|Port 8501| UI[Streamlit UI Container]
    UI -->|Internal Network: http://api:8000| API[FastAPI Microservice Container]
    HostStorage[(Host ./checkpoints)] -.->|Read-Only Volume Mount| API
```

---

## 3. Operational Status & Runtime Verification Audit

### A. Environment & Process Inspection
* **Host Platform**: macOS (Darwin 24.3.0, Apple Silicon M4)
* **Docker CLI Version**: Docker CLI v29.3.1 installed at `/opt/homebrew/bin/docker`
* **Docker Desktop GUI Application**: Detected active in process table (`/Applications/Docker.app/Contents/MacOS/Docker Desktop`, PID 40368)
* **Socket Paths**:
  - `/Users/safalyareddy/.docker/run/docker.sock` (Socket file present)
  - `/var/run/docker.sock` (Symlink pointing to `/Users/safalyareddy/.docker/run/docker.sock`)
* **Docker Context**: `desktop-linux` (`unix:///Users/safalyareddy/.docker/run/docker.sock`)

### B. Daemon Socket Responsiveness & Build Attempt Tests
To verify live daemon connectivity and container image building, empirical tests were executed against the active Docker environment:

1. **Initial Socket Probe**:
   ```bash
   curl --max-time 5 -s --unix-socket /Users/safalyareddy/.docker/run/docker.sock http://localhost/_ping
   ```
   * **Result**: Connection timed out after 5.0 seconds (curl exit code 28). Underlying LinuxKit proxy `initd.libproxy` dropped host connections.

2. **Container Build Progression (`docker compose build api`)**:
   - Docker build executed through context transfer (129.3 kB), base layer pull (`python:3.9-slim`), dependency installation (`pip install -r requirements.txt`, 557.1s), and application file copy.
   - At image export step (`exporting to image ... exporting layers 24.3s`), host physical disk reached 99.9% capacity with only **233 MiB** remaining (`/dev/disk3s1s1`).

3. **LinuxKit Virtual Machine Kernel Log (`console.log`)**:
   Inspecting `/Users/safalyareddy/Library/Containers/com.docker.docker/Data/log/vm/console.log` revealed physical APFS block allocation failure resulting in EXT4 kernel I/O errors (`-EIO`):
   ```text
   [ 9752.435773] EXT4-fs (vda1): failed to convert unwritten extents to written extents -- potential data loss!  (inode 54125, error -5)
   [ 9770.667777] EXT4-fs (vda1): failed to convert unwritten extents to written extents -- potential data loss!  (inode 24064, error -5)
   [ 9776.032702] EXT4-fs (vda1): failed to convert unwritten extents to written extents -- potential data loss!  (inode 24064, error -5)
   ```
   * Error `-5` corresponds to Linux kernel `-EIO` (I/O error) when the host hypervisor fails to grow `/Users/safalyareddy/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw`.

4. **Host Disk Pressure Verification**:
   ```bash
   df -h /
   Filesystem        Size    Used   Avail Capacity iused ifree %iused  Mounted on
   /dev/disk3s1s1   228Gi    12Gi   233Mi    99%    459k  2.4M   16%   /
   ```

### C. Formal Audit Verdict & Operational Requirements
* **Static Container Configuration**: **PASS**
  - Syntactically valid Dockerfiles for both services (`docker/Dockerfile.api` and `docker/Dockerfile.ui`).
  - Hardened `.dockerignore` blocking ~2.2 GB checkpoints and local caches from context bloat.
  - Read-only volume mount configured for checkpoints in `docker-compose.yml`.
* **Container Runtime Execution**: **BLOCKED BY HOST PHYSICAL DISK EXHAUSTION**
  - Building and running containerized ML workloads requires at least **5–10 GB** of free host disk space to store PyTorch container layers.
  - With host storage at 233 MiB, virtual disk extent conversion fails with kernel I/O errors.
  - **Host Operator Remediation**: Free 10+ GB on macOS host root partition and restart Docker Desktop to execute container runtime integration tests. Static configuration is fully hardened and production-ready.

