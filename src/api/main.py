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

# Model Cache
_MODEL_CACHE: Dict[str, Any] = {}


def get_cached_model(model_name: str = "unet_plusplus") -> Any:
    """Retrieve or instantiate model instance with state dict."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model architecture '{model_name}'. Supported: {SUPPORTED_MODELS}")

    if model_name not in _MODEL_CACHE and HAS_TORCH:
        model = build_model(model_name=model_name)
        ckpt_map = {
            "unet_plusplus": "checkpoints/unet_plusplus_best.pth",
            "deeplabv3_plus": "checkpoints/deeplabv3_plus_best.pth",
            "segformer": "checkpoints/segformer_best.pth"
        }
        ckpt_path = Path(ckpt_map.get(model_name, ""))
        if ckpt_path.exists():
            try:
                state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                try:
                    model.load_state_dict(state, strict=True)
                except Exception:
                    model.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint '{ckpt_path}': {e}")
        model.eval()
        _MODEL_CACHE[model_name] = model

    return _MODEL_CACHE.get(model_name)


@app.get("/")
def read_root():
    """API Root endpoint returning service metadata."""
    return {
        "service": "GeoVision AI — Water Body Segmentation API",
        "status": "healthy",
        "version": "1.0.0",
        "supported_models": SUPPORTED_MODELS
    }


@app.get("/health")
def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "ok",
        "torch_available": HAS_TORCH,
        "loaded_models": list(_MODEL_CACHE.keys())
    }


@app.get("/models")
def list_models():
    """List available model architectures and benchmark specs."""
    return {
        "supported_models": SUPPORTED_MODELS,
        "default_model": "unet_plusplus"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("unet_plusplus"),
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
    try:
        image = Image.open(buffer)
        image.verify()  # Verify image integrity
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupt image payload. Could not decode image."
        )

    orig_w, orig_h = image.size

    try:
        # Preprocess tensor
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        img_tensor = transform(image).unsqueeze(0)

        # Inference
        start_time = time.perf_counter()
        model = get_cached_model(model_name)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model architecture '{model_name}' could not be initialized."
            )

        with torch.no_grad():
            logits = model(img_tensor)
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

        return {
            "model_name": model_name,
            "threshold": threshold,
            "water_coverage_percentage": round(water_coverage_pct, 2),
            "water_pixels": water_pixels,
            "total_pixels": total_pixels,
            "latency_ms": round(latency_ms, 2),
            "image_dimensions": [orig_w, orig_h],
            "mask_base64": mask_b64
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference failure while processing satellite imagery."
        )
