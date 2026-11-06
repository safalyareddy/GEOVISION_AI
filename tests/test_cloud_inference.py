"""
GeoVision AI Satellite Water Body Segmentation Platform
Unit & Integration Tests for Streamlit Cloud Inference Pipeline & Backend Handling
"""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from app import (
    get_cached_pytorch_model,
    run_pytorch_inference,
    get_cached_onnx_session,
    run_onnx_inference,
    create_overlay,
)
from src.utils.model_registry import (
    CHAMPION_MODEL,
    load_verified_model,
    DEFAULT_HF_REPO,
    DEFAULT_HF_REVISION,
)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for inference tests")
def test_pytorch_inference_on_bundled_sample():
    """Verify genuine PyTorch FPN inference on the bundled patch_0047.png sample."""
    sample_path = Path("assets/demo/patch_0047.png")
    assert sample_path.exists(), "Bundled demo sample patch_0047.png must exist"

    image = Image.open(sample_path).convert("RGB")
    model = get_cached_pytorch_model(CHAMPION_MODEL)
    assert not model.training, "Model must be in evaluation mode"

    # Inference at default benchmark threshold 0.50
    mask_img, binary_mask, latency_ms, water_pct, water_area_km2 = run_pytorch_inference(
        model=model,
        image=image,
        threshold=0.50
    )

    assert mask_img.size == image.size
    assert binary_mask.shape == (256, 256)
    assert latency_ms > 0
    # Expected water coverage on patch_0047.png is ~12.63% at threshold 0.50
    assert np.isclose(water_pct, 12.63, atol=0.2)
    assert np.isclose(water_area_km2, 0.8276, atol=0.02)

    # Inference at high-precision threshold 0.60
    mask_60, _, _, pct_60, area_60 = run_pytorch_inference(
        model=model,
        image=image,
        threshold=0.60
    )
    # Calibrated threshold 0.60 yields 12.38% water coverage
    assert np.isclose(pct_60, 12.38, atol=0.2)
    assert np.isclose(area_60, 0.8115, atol=0.02)


def test_overlay_generation():
    """Verify cyan overlay generation has matching dimensions."""
    sample_path = Path("assets/demo/patch_0047.png")
    image = Image.open(sample_path).convert("RGB")
    mask = Image.new("L", image.size, color=255)

    overlay = create_overlay(image, mask)
    assert overlay.size == image.size
    assert overlay.mode == "RGB"


def test_onnx_inference_on_bundled_sample():
    """Verify genuine ONNX Runtime inference on the bundled patch_0047.png sample."""
    sample_path = Path("assets/demo/patch_0047.png")
    image = Image.open(sample_path).convert("RGB")

    sess = get_cached_onnx_session(CHAMPION_MODEL)
    assert sess is not None

    mask_img, binary_mask, latency_ms, water_pct, water_area_km2 = run_onnx_inference(
        session=sess,
        image=image,
        threshold=0.50
    )

    assert mask_img.size == image.size
    assert binary_mask.shape == (256, 256)
    assert latency_ms > 0
    assert np.isclose(water_pct, 12.63, atol=0.2)
    assert np.isclose(water_area_km2, 0.8276, atol=0.02)


def test_no_prediction_metrics_after_load_failure(tmp_path: Path):
    """Verify that if model loading fails, no prediction or mask is generated."""
    from unittest.mock import patch
    from src.utils.model_registry import MODEL_REGISTRY

    with patch.dict(MODEL_REGISTRY, {"fpn": {
        "display_name": "Broken FPN",
        "is_champion": True,
        "default_filename": "broken.pth",
        "expected_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "candidate_paths": [tmp_path / "broken.pth"],
        "hf_filename": "broken.pth"
    }}):
        with pytest.raises(RuntimeError):
            load_verified_model("fpn", device="cpu")
