"""
GeoVision AI Satellite Water Body Segmentation Platform
FastAPI Microservice for Satellite Image Segmentation Inference
"""

import io
import base64
import time
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image
import numpy as np

try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    import torch
    import torchvision.transforms as T
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models import build_model, SUPPORTED_MODELS
from src.utils.model_registry import (
    load_verified_model,
    is_model_available,
    get_model_status_summary,
    CHAMPION_MODEL,
    MODEL_REGISTRY,
)
from src.utils.input_validator import (
    classify_input_image,
    calculate_water_area_km2,
    InputClassification,
)

# Initialize FastAPI application
app = FastAPI(
    title="GeoVision AI — Water Body Segmentation API",
    description="High-performance machine learning inference API for satellite imagery water body detection.",
    version="1.0.0"
)

# Enable CORS for frontend integration
if HAS_FASTAPI:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Model Cache and Runtime Metrics
_MODEL_CACHE: Dict[str, Any] = {}
_METRICS: Dict[str, Any] = {
    "total_requests": 0,
    "successful_inferences": 0,
    "rejected_photos": 0,
    "errors": 0,
    "requests_by_model": {m: 0 for m in SUPPORTED_MODELS},
}


def get_cached_model(model_name: str = CHAMPION_MODEL) -> Any:
    """Retrieve or instantiate verified model instance with state dict."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model architecture '{model_name}'. Supported: {SUPPORTED_MODELS}")

    if model_name not in _MODEL_CACHE and HAS_TORCH:
        # load_verified_model verifies SHA-256 and strictly disallows uninitialized execution
        model = load_verified_model(model_name=model_name, device="cpu")
        _MODEL_CACHE[model_name] = model

    return _MODEL_CACHE.get(model_name)


@app.get("/")
def read_root():
    """API Root endpoint returning service metadata."""
    return {
        "service": "GeoVision AI — Water Body Segmentation API",
        "status": "healthy",
        "version": "1.0.0",
        "supported_models": SUPPORTED_MODELS,
        "champion_model": CHAMPION_MODEL
    }


@app.get("/health")
def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "ok",
        "torch_available": HAS_TORCH,
        "champion_model": CHAMPION_MODEL,
        "champion_ready": is_model_available(CHAMPION_MODEL),
        "loaded_models": list(_MODEL_CACHE.keys())
    }


@app.get("/ready")
def readiness_check():
    """
    Truthful readiness reporting.
    Reports API readiness state without claiming uninstantiated models are loaded.
    """
    champion_ready = is_model_available(CHAMPION_MODEL)
    status_summary = get_model_status_summary()
    available_count = sum(1 for v in status_summary.values() if v.get("available"))

    return {
        "ready": champion_ready and HAS_TORCH,
        "service_state": "READY" if champion_ready else "DEGRADED",
        "torch_available": HAS_TORCH,
        "champion_model": CHAMPION_MODEL,
        "champion_available": champion_ready,
        "loaded_models_in_memory": list(_MODEL_CACHE.keys()),
        "total_available_models": f"{available_count}/{len(SUPPORTED_MODELS)}",
        "lazy_loading_policy": "Non-cached models are lazily loaded and cryptographically verified on first predict request."
    }


@app.get("/models")
def list_models():
    """List available model architectures and benchmark specs."""
    return {
        "supported_models": SUPPORTED_MODELS,
        "default_model": CHAMPION_MODEL,
        "champion_model": CHAMPION_MODEL,
        "models_status": get_model_status_summary()
    }


@app.get("/models/{model_id}/status")
def get_model_status(model_id: str):
    """Retrieve detailed verification and loading status for a specific architecture."""
    if model_id not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model architecture '{model_id}' is not registered. Choose from: {list(SUPPORTED_MODELS.keys())}"
        )
    summary = get_model_status_summary()
    model_info = summary.get(model_id, {})
    model_info["is_cached_in_memory"] = model_id in _MODEL_CACHE
    return model_info


@app.get("/metrics")
def get_metrics():
    """Retrieve inference operational metrics."""
    return {
        "service": "GeoVision AI Inference Service",
        "metrics": _METRICS,
        "loaded_models_count": len(_MODEL_CACHE),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(CHAMPION_MODEL),
    threshold: float = Form(0.5)
):
    """
    Predict surface water segmentation mask for uploaded satellite image file.

    Returns water coverage percentage, inference time, and base64 PNG mask string.
    """
    if not HAS_TORCH:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PyTorch is unavailable on the server."
        )

    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model '{model_name}'. Choose from: {SUPPORTED_MODELS}"
        )

    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Threshold must be a float between 0.0 and 1.0."
        )

    # Validate uploaded content type
    allowed_content_types = ["image/png", "image/jpeg", "image/jpg", "image/tiff", "application/octet-stream"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image file format '{file.content_type}'."
        )

    # Enforce streaming size limit (10 MB max)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    chunk_size = 65536
    total_read = 0
    buffer = io.BytesIO()

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_read += len(chunk)
        if total_read > MAX_FILE_SIZE:
            status_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
            raise HTTPException(
                status_code=status_413,
                detail="Uploaded file exceeds maximum size limit of 10 MB."
            )
        buffer.write(chunk)

    if total_read == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty (0 bytes)."
        )

    buffer.seek(0)
    raw_bytes = buffer.getvalue()
    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.verify()  # Verify image integrity
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        _METRICS["errors"] += 1
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupt image payload. Could not decode image."
        )

    # Scientific Safety Gate: Classify image
    classification, input_meta = classify_input_image(image, raw_bytes=raw_bytes, filename=file.filename)
    if classification == InputClassification.UNSUPPORTED_PHOTOGRAPH:
        _METRICS["rejected_photos"] += 1
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=input_meta["rejection_reason"]
        )
    elif classification == InputClassification.INVALID_IMAGE:
        _METRICS["errors"] += 1
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=input_meta["rejection_reason"]
        )

    orig_w, orig_h = image.size

    try:
        # Preprocess tensor (matching standard [0, 1] training normalization)
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        img_tensor = transform(image).unsqueeze(0)

        # Inference
        start_time = time.perf_counter()
        model = get_cached_model(model_name)
        if model is None:
            _METRICS["errors"] += 1
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model architecture '{model_name}' could not be initialized."
            )

        with torch.inference_mode():
            out = model(img_tensor)
            if isinstance(out, dict):
                logits = out.get("out", out.get("logits"))
            elif hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Postprocess mask
        binary_mask = (probs >= threshold).astype(np.uint8) * 255
        water_pixels = int(np.sum(binary_mask > 0))
        total_pixels = int(binary_mask.size)
        water_coverage_pct = float((water_pixels / total_pixels) * 100.0)

        # Resize mask back to original input dimensions
        mask_img = Image.fromarray(binary_mask, mode="L").resize((orig_w, orig_h), Image.NEAREST)

        # Convert mask image to base64
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Geospatial & Scientific Area Quantification
        area_km2, area_note = calculate_water_area_km2(water_pixels, classification, input_meta)

        # Probability Distribution Diagnostics
        prob_stats = {
            "min": round(float(np.min(probs)), 4),
            "max": round(float(np.max(probs)), 4),
            "mean": round(float(np.mean(probs)), 4),
            "std": round(float(np.std(probs)), 4),
            "is_saturated": bool(np.std(probs) < 0.01),
        }

        # Update metrics
        _METRICS["total_requests"] += 1
        _METRICS["successful_inferences"] += 1
        _METRICS["requests_by_model"][model_name] = _METRICS["requests_by_model"].get(model_name, 0) + 1

        return {
            "model_name": model_name,
            "threshold": threshold,
            "input_classification": classification.value,
            "water_coverage_percentage": round(water_coverage_pct, 2),
            "estimated_water_area_km2": area_km2,
            "area_estimation_basis": area_note,
            "water_pixels": water_pixels,
            "total_pixels": total_pixels,
            "latency_ms": round(latency_ms, 2),
            "probability_stats": prob_stats,
            "image_dimensions": [orig_w, orig_h],
            "mask_base64": mask_b64
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        if isinstance(e, RuntimeError):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference failure while processing satellite imagery."
        )
