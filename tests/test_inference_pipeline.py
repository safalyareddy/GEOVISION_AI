"""
GeoVision AI Satellite Water Body Segmentation Platform
Unit & Integration Tests for Unified Direct Inference Pipeline (PyTorch & ONNX Runtime)
"""

import io
import pytest
import numpy as np
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from src.models import SUPPORTED_MODELS
from src.inference import (
    predict,
    predict_onnx,
    predict_all,
    load_model,
    load_onnx_session,
    get_model_metadata,
    get_model_status,
)
from src.utils.model_registry import (
    CHAMPION_MODEL,
    is_model_available,
    get_model_status_summary,
)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for inference pipeline tests.")
class TestInferencePipeline:
    """Verifies direct in-process inference across all 4 models and 2 execution engines."""

    @pytest.mark.parametrize("model_name", list(SUPPORTED_MODELS.keys()))
    def test_pytorch_inference_all_models(self, model_name: str):
        """Verify direct PyTorch inference produces valid mask, probability map, and coverage."""
        img = Image.new("RGB", (128, 128), color=(60, 75, 70))
        res = predict(image=img, model_name=model_name, threshold=0.50, engine="pytorch")

        assert res["model_name"] == model_name
        assert res["engine"] == "pytorch"
        assert res["latency_ms"] > 0
        assert 0.0 <= res["water_coverage_pct"] <= 100.0
        assert res["mask_array_256"].shape == (256, 256)
        assert res["probability_map_256"].shape == (256, 256)
        assert np.all((res["probability_map_256"] >= 0.0) & (res["probability_map_256"] <= 1.0))
        assert res["mask_image"].size == (128, 128)
        assert res["binary_mask"].shape == (128, 128)

    @pytest.mark.skipif(not HAS_ONNX, reason="ONNX Runtime required for ONNX pipeline tests.")
    @pytest.mark.parametrize("model_name", list(SUPPORTED_MODELS.keys()))
    def test_onnx_inference_all_models(self, model_name: str):
        """Verify direct ONNX Runtime inference produces valid mask, probability map, and coverage."""
        img = Image.new("RGB", (128, 128), color=(60, 75, 70))
        res = predict_onnx(image=img, model_name=model_name, threshold=0.50)

        assert res["model_name"] == model_name
        assert res["engine"] == "onnx"
        assert res["latency_ms"] > 0
        assert 0.0 <= res["water_coverage_pct"] <= 100.0
        assert res["mask_array_256"].shape == (256, 256)
        assert res["probability_map_256"].shape == (256, 256)
        assert np.all((res["probability_map_256"] >= 0.0) & (res["probability_map_256"] <= 1.0))
        assert res["mask_image"].size == (128, 128)
        assert res["binary_mask"].shape == (128, 128)

    def test_invalid_model_name_raises_value_error(self):
        """Verify unsupported model identifier raises ValueError."""
        img = Image.new("RGB", (64, 64), color=(60, 75, 70))
        with pytest.raises(ValueError, match="Unknown architecture"):
            predict(image=img, model_name="nonexistent_model")

    def test_invalid_threshold_raises_value_error(self):
        """Verify threshold outside [0, 1] raises ValueError."""
        img = Image.new("RGB", (64, 64), color=(60, 75, 70))
        with pytest.raises(ValueError, match="Threshold must be in"):
            predict(image=img, model_name=CHAMPION_MODEL, threshold=1.5)
        with pytest.raises(ValueError, match="Threshold must be in"):
            predict(image=img, model_name=CHAMPION_MODEL, threshold=-0.1)

    def test_predict_all_models_sequential(self):
        """Verify sequential inference across all available models."""
        img = Image.new("RGB", (64, 64), color=(60, 75, 70))
        results = predict_all(image=img, threshold=0.50, engine="pytorch")
        for m_name in SUPPORTED_MODELS.keys():
            assert m_name in results
            assert "water_coverage_pct" in results[m_name]

    def test_metadata_and_status(self):
        """Verify metadata and status reporting for all supported models."""
        for m_name in SUPPORTED_MODELS.keys():
            meta = get_model_metadata(m_name)
            assert meta["model_name"] == m_name
            assert meta["is_champion"] == (m_name == CHAMPION_MODEL)
            status = get_model_status(m_name)
            assert status["available"] is True
