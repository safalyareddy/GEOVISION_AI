# Security Architecture & Hardening Controls

---

## 1. Threat Modeling & Defensive Protections

### A. Checkpoint Deserialization Hardening
Standard Python `pickle` loading carries remote code execution vulnerability risks. All weight loading calls in GeoVision AI explicitly enforce:
```python
state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
```
This restricts deserialization strictly to tensor data, completely preventing arbitrary code execution.

### B. Upload Denial-of-Service Defense
* **Streaming Chunks**: File uploads are processed in 64 KB chunks up to an absolute ceiling of 10 MB.
* **Early Abort**: If uploaded payload exceeds 10 MB, reading is aborted and `HTTP 413` is returned immediately without buffering the remainder to RAM.
* **Integrity Validation**: Executes `PIL.Image.open().verify()` prior to decompression, discarding zip bombs or malformed binaries.

### C. Secret Isolation
* Verified 0 API keys or private tokens committed across all branches and historical commits.
* `.gitignore` explicitly blocks `.env*`, `*.key`, `*.pem`, `*.crt`, and archive bundles (`*.tar.gz`).
* Safe template provided in `.env.example`.
