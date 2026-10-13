# Final Docker Runtime & Deployment Audit

**Audit Date**: October 13, 2026  
**Auditor**: Infrastructure & DevOps Systems Engineer  
**Scope**: Container Definitions, Dockerfile Analysis, Volume Mounts, Host Runtime Constraints

---

## 1. Static Configuration Verification

* **API Service Dockerfile** (`docker/Dockerfile.api`):
  - Base Image: `python:3.9-slim`
  - System Dependencies: `libgl1`, `libglib2.0-0` installed cleanly with apt cache cleanup (`rm -rf /var/lib/apt/lists/*`).
  - Application Directory: `/app`
  - Port Expose: `8000`
  - Entrypoint: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
  - **Verdict**: **PASS — VERIFIED**
* **UI Service Dockerfile** (`docker/Dockerfile.ui`):
  - Base Image: `python:3.9-slim`
  - System Dependencies: `libgl1`, `libglib2.0-0` installed cleanly.
  - Entrypoint: `streamlit run app.py --server.port 8501 --server.address 0.0.0.0` (Corrected from legacy `client/app.py`).
  - **Verdict**: **PASS — VERIFIED**
* **Context Isolation** (`.dockerignore`):
  - Strictly excludes `.git`, `checkpoints/`, `dataset/`, `venv/`, `__pycache__`, and scratch files.
  - Reduces build context transfer from 2.2+ GB down to ~129 KB.
  - **Verdict**: **PASS — VERIFIED**
* **Docker Compose Orchestration** (`docker-compose.yml`):
  - Services: `api` (port 8000) and `ui` (port 8501).
  - Checkpoint Mount: `./checkpoints:/app/checkpoints:ro` (read-only attribute prevents container corruption of host weights).
  - Environment: `API_URL=http://api:8000` configured for inter-container communication.
  - **Verdict**: **PASS — VERIFIED**

---

## 2. Empirical Runtime Failure & Host Hardware Constraint

* **Audit Verdict**: **BLOCKED — NOT VERIFIED**

### Empirical Evidence & Reproduction Details:
1. **Host Disk Pressure**:
   ```bash
   df -h /
   Filesystem        Size    Used   Avail Capacity iused ifree %iused  Mounted on
   /dev/disk3s1s1   228Gi    12Gi   947Mi    93%    459k  9.7M    5%   /
   ```
   - Physical host storage has only **~947 MiB** free.
   - Building and storing uncompressed PyTorch Docker images (`torch`, `torchvision`, `transformers`, `scipy`) requires a minimum of **5–10 GB** of free host disk space.
2. **Build Progression & Point of Failure**:
   - `docker compose build api` progressed through context transfer (129.3 kB), base layer pull (`python:3.9-slim`), and pip installation (`pip install -r requirements.txt`, 557.1s).
   - At the layer export phase (`exporting to image ... exporting layers 24.3s`), host disk allocation failed.
3. **LinuxKit Hypervisor VM Crash Evidence (`console.log`)**:
   Inspecting `/Users/safalyareddy/Library/Containers/com.docker.docker/Data/log/vm/console.log` revealed physical block extent allocation failure leading to EXT4 filesystem I/O errors (`-EIO`):
   ```text
   [ 9752.435773] EXT4-fs (vda1): failed to convert unwritten extents to written extents -- potential data loss!  (inode 54125, error -5)
   [ 9770.667777] EXT4-fs (vda1): failed to convert unwritten extents to written extents -- potential data loss!  (inode 24064, error -5)
   [ 9776.032702] EXT4-fs (vda1): failed to convert unwritten extents to written extents -- potential data loss!  (inode 24064, error -5)
   ```
4. **Socket Timeout**:
   - The LinuxKit VM proxy dropped all host socket requests:
     ```bash
     curl --max-time 5 -s --unix-socket /Users/safalyareddy/.docker/run/docker.sock http://localhost/_ping
     # Result: curl exit code 28 (Connection timed out after 5.0s)
     ```

### Remediation Protocol for Host Operator:
To unblock containerized testing:
1. Free at least 15 GB of host disk space on macOS APFS root `/`.
2. Fully quit and restart Docker Desktop.
3. Execute `docker compose up --build -d` and run `curl http://localhost:8000/health`.
