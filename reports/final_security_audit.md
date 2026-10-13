# Final Security, Supply-Chain & Deserialization Audit

**Audit Date**: October 13, 2026  
**Auditor**: DevSecOps & Security Specialist  
**Scope**: Secrets in Git History, Safe Deserialization, API Payload Protections, Supply-Chain Dependencies

---

## 1. Secrets & Credential Exposure Scan

* **Method**: Recursive automated pattern scan across active working tree and commit history for private keys, AWS tokens, GitHub personal access tokens, database credentials, and `.env` files.
* **Findings**:
  - Committed `.env` files: **0** (strictly ignored via `.gitignore`).
  - Active credentials / tokens in code: **0**.
  - GitHub Personal Access Tokens (`gho_` / `ghp_`): Sanitized, keyring-based authentication only.
* **Verdict**: **PASS — VERIFIED**

---

## 2. Safe Model Deserialization (`weights_only=True`)

* **Vulnerability Mitigated**: Arbitrary code execution via Python `pickle` deserialization during checkpoint loading.
* **Audit of all `torch.load` invocations**:
  1. `src/api/main.py:67`: `torch.load(..., weights_only=True)` — **ENFORCED**
  2. `src/evaluate.py:201`: `torch.load(..., weights_only=True)` — **ENFORCED**
  3. `app.py:66`: `torch.load(..., weights_only=True)` — **ENFORCED**
  4. `scripts/debug_inference.py:64`: `torch.load(..., weights_only=True)` — **ENFORCED**
  5. `scripts/export_onnx.py:66`: Updated to enforce `weights_only=True` (Fixed in Section 8 defect resolution).
* **Verdict**: **PASS — VERIFIED**

---

## 3. API Payload & Input Validation Hardening

* **Streaming Upload Limit (10 MB)**:
  - Enforced via chunked streaming counter (`chunk_size=65536`) inside `predict()` endpoint before loading full image buffer into memory.
  - Rejects oversized payloads with HTTP 413 (`HTTP_413_CONTENT_TOO_LARGE`).
* **MIME Whitelisting**:
  - Restricts uploads strictly to `image/png`, `image/jpeg`, `image/jpg`, `image/tiff`, `application/octet-stream`. Rejects text, scripts, or binaries with HTTP 400.
* **Image Header Integrity Check**:
  - Calls `Image.verify()` on PIL buffer prior to decoding to prevent decompression bombs and memory corruption exploits.
* **Verdict**: **PASS — VERIFIED**

---

## 4. Supply-Chain Dependencies & Realistic Limitations

* **Dependency Manifest**: `requirements.txt` specifies explicit package versions.
* **Security Disclaimer**:
  - While static scans detect zero current high-severity CVEs, automated audits cannot provide a mathematical guarantee of "zero risk".
  - Complex upstream machine learning dependencies (`torch`, `transformers`, `scipy`) contain native C++ binaries that require periodic vulnerability scanning and dependency updates.
* **Verdict**: **WARNING — LIMITATION REMAINS** (Standard third-party dependency supply-chain surface).
