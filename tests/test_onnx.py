"""
GeoVision AI Satellite Water Body Segmentation Platform
C11 Unit Tests: ONNX Model Export and Inference Engine
"""

from pathlib import Path
import pytest
import numpy as np

from scripts.export_onnx import export_to_onnx
from scripts.debug_inference import compare_pytorch_and_onnx_inference

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_export_to_onnx_tmp_path(tmp_path):
    """Test ONNX export pipeline writing to temporary directory."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available for ONNX export test.")

    out_file = tmp_path / "test_model.onnx"
    success = export_to_onnx(
        model_name="unet_plusplus",
        checkpoint_path=None,
        output_path=str(out_file),
        img_size=32,
        batch_size=1
    )

    assert success is True
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_compare_pytorch_and_onnx_inference_missing_onnx_file(tmp_path):
    """Test inference comparison gracefully handles missing ONNX file."""
    non_existent = tmp_path / "non_existent.onnx"
    res = compare_pytorch_and_onnx_inference(
        model_name="unet_plusplus",
        onnx_path=str(non_existent),
        img_size=32,
        num_runs=2
    )

    assert "model_name" in res
    assert res["model_name"] == "unet_plusplus"
