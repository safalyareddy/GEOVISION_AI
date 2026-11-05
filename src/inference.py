"""
GeoVision AI — Unified Production PyTorch Inference Pipeline

Provides standardized, safe, multi-architecture in-process inference for:
- FPN–ResNet34 (Champion)
- UNet++–ResNet34 (Baseline/Improved)
- DeepLabV3+–ResNet34 (Improved)
- SegFormer–MiT-B0 (Lightweight Vision Transformer)

Adheres strictly to:
- RGB float32 [0.0, 1.0] preprocessing parity
- Explicit device management (CPU and Apple Silicon MPS)
- torch.inference_mode() and model.eval() guarantees
- Re-use of loaded model instances (zero redundant disk loads)
- Strict cryptographic checkpoint verification before execution
"""

import io
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from PIL import Image
import numpy as np

try:
    import torch
    import torchvision.transforms as T
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models import SUPPORTED_MODELS
from src.utils.model_registry import (
    load_verified_model,
    get_model_status_summary,
    MODEL_REGISTRY,
    CHAMPION_MODEL,
    is_model_available,
)

# Global in-process model cache
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}

# Standard Spatial Resolution & Nominal Sentinel-2 Ground Sample Distance
STANDARD_INPUT_SIZE = (256, 256)
NOMINAL_GSD_METERS = 10.0
PIXEL_AREA_KM2 = (NOMINAL_GSD_METERS * NOMINAL_GSD_METERS) / 1e6  # 0.0001 km^2 per pixel


def get_optimal_device(prefer_mps: bool = False) -> str:
    """Return 'mps' if Apple Silicon GPU is available and requested, else 'cpu'."""
    if not HAS_TORCH:
        return "cpu"
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str = CHAMPION_MODEL, device: str = "cpu") -> Any:
    """
    Load and cache a verified PyTorch model instance in evaluation mode.
    Guarantees strict state-dict loading and integrity checks.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for model inference.")

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown architecture '{model_name}'. Choose from: {list(SUPPORTED_MODELS.keys())}")

    device = device.lower()
    cache_key = (model_name, device)

    if cache_key not in _MODEL_CACHE:
        # Load verified model via registry (checks SHA-256 and strict weights match)
        model = load_verified_model(model_name=model_name, device=device)
        model.eval()
        _MODEL_CACHE[cache_key] = model

    return _MODEL_CACHE[cache_key]


def clear_model_cache() -> None:
    """Clear in-memory cached model weights."""
    _MODEL_CACHE.clear()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model_metadata(model_name: str) -> Dict[str, Any]:
    """Retrieve verified engineering metadata for a registered model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry.")

    meta = MODEL_REGISTRY[model_name]
    summary = get_model_status_summary().get(model_name, {})

    # Compute parameter counts based on architecture
    param_counts = {
        "fpn": 23155393,
        "unet_plusplus": 26078609,
        "deeplabv3_plus": 22437457,
        "segformer": 3714401,
    }

    encoders = {
        "fpn": "resnet34",
        "unet_plusplus": "resnet34",
        "deeplabv3_plus": "resnet34",
        "segformer": "nvidia/mit-b0",
    }

    return {
        "model_name": model_name,
        "display_name": meta.get("display_name", model_name),
        "architecture": model_name,
        "encoder": encoders.get(model_name, "resnet34"),
        "framework": "PyTorch",
        "is_champion": meta.get("is_champion", False),
        "expected_sha256": meta.get("expected_sha256"),
        "checkpoint_path": summary.get("checkpoint_path"),
        "size_mb": meta.get("size_mb"),
        "parameter_count": param_counts.get(model_name),
        "input_spatial_dimensions": [256, 256],
        "input_channels": 3,
        "normalization": "RGB float32 scaled to [0.0, 1.0]",
        "output_activation": "Sigmoid",
        "default_threshold": 0.50,
        "available": summary.get("available", False),
        "support_note": meta.get("support_note", ""),
    }


def get_model_status(model_name: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Return model status summary including cache and availability state."""
    summary = get_model_status_summary()
    for name, item in summary.items():
        item["in_cache"] = any(k[0] == name for k in _MODEL_CACHE.keys())

    if model_name:
        if model_name not in summary:
            raise ValueError(f"Model '{model_name}' not in registry.")
        return summary[model_name]
    return summary


def _prepare_image_input(image: Union[Image.Image, np.ndarray, str, Path]) -> Tuple[Image.Image, int, int]:
    """Normalize input into PIL RGB image while recording original dimensions."""
    if isinstance(image, (str, Path)):
        p = Path(image)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found at: {p}")
        pil_img = Image.open(p).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_img = Image.fromarray(image).convert("RGB")
        elif image.ndim == 3:
            if image.shape[2] == 4:
                pil_img = Image.fromarray(image, mode="RGBA").convert("RGB")
            elif image.shape[2] == 3:
                pil_img = Image.fromarray(image, mode="RGB")
            elif image.shape[0] in [1, 3]:
                # CHW format
                arr = np.transpose(image, (1, 2, 0))
                pil_img = Image.fromarray(arr).convert("RGB")
            else:
                raise ValueError(f"Unsupported image array shape: {image.shape}")
        else:
            raise ValueError(f"Invalid image array dimensions: {image.shape}")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    orig_w, orig_h = pil_img.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Image has invalid spatial dimensions ({orig_w}x{orig_h}). Dimensions must be positive.")
    return pil_img, orig_w, orig_h


def predict(
    image: Union[Image.Image, np.ndarray, str, Path],
    model_name: str = CHAMPION_MODEL,
    threshold: float = 0.50,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute high-fidelity PyTorch surface water segmentation inference.

    Args:
        image: PIL Image, NumPy array, or path to image file.
        model_name: Registered model identifier ('fpn', 'unet_plusplus', 'deeplabv3_plus', 'segformer').
        threshold: Binary decision probability cutoff in [0.0, 1.0].
        device: Target execution device ('cpu', 'mps'). Defaults to 'cpu'.

    Returns:
        Dict containing predicted mask array, PIL mask, coverage, area, latency, and metadata.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Threshold must be in [0.0, 1.0], got {threshold}")

    target_device = device or "cpu"
    pil_img, orig_w, orig_h = _prepare_image_input(image)

    # 1. Transform: PIL -> Resize (256, 256) -> Float32 [0.0, 1.0] -> NCHW tensor
    transform = T.Compose([
        T.Resize(STANDARD_INPUT_SIZE),
        T.ToTensor()
    ])
    input_tensor = transform(pil_img).unsqueeze(0).to(target_device)

    # 2. Model retrieval
    model = load_model(model_name=model_name, device=target_device)

    # 3. Deterministic Forward Pass
    start_time = time.perf_counter()
    with torch.inference_mode():
        out = model(input_tensor)
        if isinstance(out, dict):
            logits = out.get("out", out.get("logits"))
        elif hasattr(out, "logits"):
            logits = out.logits
        else:
            logits = out

        # Compute probability map via Sigmoid
        probs_tensor = torch.sigmoid(logits).squeeze(0).squeeze(0)
        probs_256 = probs_tensor.cpu().numpy()

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # 4. Binary Thresholding
    binary_mask_256 = (probs_256 >= threshold).astype(np.uint8)
    water_pixels = int(np.sum(binary_mask_256))
    total_pixels = int(binary_mask_256.size)
    water_coverage_pct = float((water_pixels / total_pixels) * 100.0)
    water_area_km2 = float(round((water_pixels * (NOMINAL_GSD_METERS ** 2)) / 1e6, 4))

    # 5. Mask at original input resolution
    mask_256_pil = Image.fromarray(binary_mask_256 * 255, mode="L")
    mask_orig_pil = mask_256_pil.resize((orig_w, orig_h), resample=Image.NEAREST)
    binary_mask_orig = (np.array(mask_orig_pil) > 127).astype(np.uint8)

    return {
        "model_name": model_name,
        "device": target_device,
        "threshold": float(threshold),
        "latency_ms": round(latency_ms, 2),
        "water_coverage_pct": round(water_coverage_pct, 2),
        "water_area_km2": water_area_km2,
        "water_pixels": water_pixels,
        "total_pixels": total_pixels,
        "original_dimensions": [orig_w, orig_h],
        "mask_array_256": binary_mask_256,
        "probability_map_256": probs_256,
        "mask_image": mask_orig_pil,
        "binary_mask": binary_mask_orig,
        "nominal_resolution": "Sentinel-2 10m GSD (100 m²/px)",
        "geospatial_disclaimer": "Unprojected nominal Sentinel-2 10m Ground Sample Distance approximation (100 m²/pixel). Not georeferenced survey grade without external GeoTIFF CRS/affine metadata."
    }


def predict_all(
    image: Union[Image.Image, np.ndarray, str, Path],
    threshold: float = 0.50,
    device: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Execute inference across all verified available models sequentially.
    """
    results = {}
    pil_img, _, _ = _prepare_image_input(image)

    for model_name in SUPPORTED_MODELS.keys():
        if is_model_available(model_name):
            try:
                res = predict(image=pil_img, model_name=model_name, threshold=threshold, device=device)
                results[model_name] = res
            except Exception as e:
                results[model_name] = {"error": str(e), "model_name": model_name}
    return results
