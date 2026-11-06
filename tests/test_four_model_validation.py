"""
GeoVision AI — Phase 5: Four-Model PyTorch Functional Test Suite

Comprehensive automated test suite covering Tests A through H across:
1. UNet++–ResNet34
2. DeepLabV3+–ResNet34
3. SegFormer–MiT-B0
4. FPN–ResNet34 (Champion)
"""

import io
import time
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

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

from src.models import build_model, SUPPORTED_MODELS
from src.utils.model_registry import (
    load_verified_model,
    compute_file_sha256,
    resolve_checkpoint_path,
    is_model_available,
    get_model_status_summary,
    MODEL_REGISTRY,
    CHAMPION_MODEL,
)
from src.inference import (
    load_model,
    predict,
    predict_all,
    get_model_metadata,
    get_model_status,
    clear_model_cache,
    STANDARD_INPUT_SIZE,
)
from app import (
    run_pytorch_inference,
    get_cached_pytorch_model,
    create_overlay,
)

EXPECTED_PARAMS = {
    "fpn": 23155393,
    "unet_plusplus": 26078609,
    "deeplabv3_plus": 22437457,
    "segformer": 3714401,
}

ALL_MODELS = ["fpn", "unet_plusplus", "deeplabv3_plus", "segformer"]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for functional tests.")
class TestFourModelValidation:
    """Rigorous verification suite for all four segmentation architectures."""

    # -------------------------------------------------------------------------
    # TEST A: Architecture Construction
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_a_architecture_construction(self, model_name: str):
        """Verify that every model can be instantiated successfully without weights."""
        model = build_model(model_name=model_name, encoder_weights=None)
        assert isinstance(model, torch.nn.Module), f"Model {model_name} is not a torch.nn.Module"
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == EXPECTED_PARAMS[model_name], (
            f"Param count mismatch for {model_name}: expected {EXPECTED_PARAMS[model_name]}, got {total_params}"
        )

    # -------------------------------------------------------------------------
    # TEST B: Checkpoint Loading & Cryptographic Integrity
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_b_checkpoint_loading(self, model_name: str):
        """Verify checkpoint existence, SHA-256 match, strict state-dict loading, and eval mode."""
        ckpt_path = resolve_checkpoint_path(model_name, allow_download=False)
        assert ckpt_path is not None and ckpt_path.exists(), f"Missing checkpoint for {model_name}"

        # SHA-256 Check
        expected_sha = MODEL_REGISTRY[model_name]["expected_sha256"]
        actual_sha = compute_file_sha256(ckpt_path)
        assert actual_sha.lower() == expected_sha.lower(), (
            f"SHA-256 mismatch for {model_name}: expected {expected_sha}, got {actual_sha}"
        )

        # Strict load via registry
        model = load_verified_model(model_name, device="cpu")
        assert not model.training, f"Model {model_name} was not set to evaluation mode"
        loaded_params = sum(p.numel() for p in model.parameters())
        assert loaded_params == EXPECTED_PARAMS[model_name]

    # -------------------------------------------------------------------------
    # TEST C: Tensor Forward Pass Inference
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_c_tensor_inference(self, model_name: str):
        """Verify forward pass with random tensor: shape, dtype, device, finite values, logits, and probs."""
        model = load_verified_model(model_name, device="cpu")
        input_tensor = torch.randn(1, 3, 256, 256, dtype=torch.float32, device="cpu")

        with torch.inference_mode():
            out = model(input_tensor)
            if isinstance(out, dict):
                logits = out.get("out", out.get("logits"))
            elif hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out

            assert logits.shape == (1, 1, 256, 256), f"Shape mismatch for {model_name}: {logits.shape}"
            assert logits.dtype == torch.float32
            assert logits.device.type == "cpu"
            assert torch.isfinite(logits).all(), f"Non-finite logits detected in {model_name}"

            probs = torch.sigmoid(logits)
            assert (probs >= 0.0).all() and (probs <= 1.0).all(), f"Probabilities outside [0, 1] for {model_name}"

    # -------------------------------------------------------------------------
    # TEST D: Segmentation Output & Mask Validity
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_d_segmentation_output(self, model_name: str):
        """Verify binary mask output, [0, 1] bounds, non-crashing execution on different image inputs."""
        # Test 1: Real satellite test image
        img_p = Path("dataset/test/images/patch_0000.png")
        if img_p.exists():
            test_img = Image.open(img_p)
        else:
            test_img = Image.new("RGB", (256, 256), color=(20, 40, 60))

        res = predict(test_img, model_name=model_name, threshold=0.50, device="cpu")
        assert res["mask_array_256"].shape == (256, 256)
        assert set(np.unique(res["mask_array_256"])).issubset({0, 1})
        assert not np.isnan(res["mask_array_256"]).any()
        assert 0.0 <= res["water_coverage_pct"] <= 100.0
        assert res["water_area_km2"] >= 0.0

        # Test 2: Solid black image
        black_img = Image.new("RGB", (128, 128), color="black")
        res_black = predict(black_img, model_name=model_name, threshold=0.50, device="cpu")
        assert res_black["mask_image"].size == (128, 128)

        # Test 3: Solid white image
        white_img = Image.new("RGB", (300, 200), color="white")
        res_white = predict(white_img, model_name=model_name, threshold=0.50, device="cpu")
        assert res_white["mask_image"].size == (300, 200)

    # -------------------------------------------------------------------------
    # TEST E: Device Execution (CPU and Apple Silicon MPS)
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_e_device_testing(self, model_name: str):
        """Verify execution on CPU and Apple Silicon MPS (when available)."""
        test_img = Image.new("RGB", (256, 256), color="navy")

        # CPU Run
        res_cpu = predict(test_img, model_name=model_name, threshold=0.50, device="cpu")
        assert res_cpu["device"] == "cpu"
        assert res_cpu["mask_array_256"].shape == (256, 256)

        # MPS Run if available
        if torch.backends.mps.is_available():
            res_mps = predict(test_img, model_name=model_name, threshold=0.50, device="mps")
            assert res_mps["device"] == "mps"
            assert res_mps["mask_array_256"].shape == (256, 256)
            # Binary masks should match identically
            np.testing.assert_array_equal(res_cpu["mask_array_256"], res_mps["mask_array_256"])

    # -------------------------------------------------------------------------
    # TEST F: Repeated Inference & State Isolation
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_f_repeated_inference(self, model_name: str):
        """Verify repeated inference: no memory corruption, no stale caches, no training-mode leakage."""
        test_img = Image.new("RGB", (256, 256), color=(45, 80, 120))
        model = load_model(model_name, device="cpu")

        initial_mask = None
        for i in range(5):
            res = predict(test_img, model_name=model_name, threshold=0.50, device="cpu")
            assert not model.training, f"Model {model_name} accidentally switched to training mode!"

            if initial_mask is None:
                initial_mask = res["mask_array_256"]
            else:
                np.testing.assert_array_equal(
                    initial_mask,
                    res["mask_array_256"],
                    err_msg=f"Inconsistent prediction on repeat iteration {i} for {model_name}"
                )

    # -------------------------------------------------------------------------
    # TEST G: ONNX Runtime Integration
    # -------------------------------------------------------------------------
    @pytest.mark.skipif(not HAS_ONNX, reason="ONNX Runtime required")
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_g_onnx_integration(self, model_name: str):
        """Verify model loading, session initialization, and prediction via ONNX Runtime."""
        from src.inference import load_onnx_session, predict_onnx

        sess = load_onnx_session(model_name)
        assert sess is not None

        # Predict with test tile
        img = Image.new("RGB", (128, 128), color=(60, 75, 70))
        res = predict_onnx(img, model_name=model_name, threshold=0.50)

        assert res["model_name"] == model_name
        assert res["engine"] == "onnx"
        assert res["latency_ms"] > 0
        assert 0.0 <= res["water_coverage_pct"] <= 100.0
        assert res["mask_array_256"].shape == (256, 256)
        assert np.all((res["probability_map_256"] >= 0.0) & (res["probability_map_256"] <= 1.0))
        assert res["mask_image"].size == (128, 128)

    # -------------------------------------------------------------------------
    # TEST H: Frontend Model Integration & Availability
    # -------------------------------------------------------------------------
    def test_h_frontend_integration(self):
        """Verify all four models are registered for frontend selection with accurate availability."""
        status_map = get_model_status_summary()
        assert len(status_map) == 4

        for m_name in ALL_MODELS:
            assert m_name in status_map
            item = status_map[m_name]
            assert item["available"] is True
            assert item["load_supported"] is True
            assert item["checkpoint_path"] is not None

        # Verify champion designation
        assert status_map["fpn"]["is_champion"] is True
        assert status_map["unet_plusplus"]["is_champion"] is False
        assert status_map["deeplabv3_plus"]["is_champion"] is False
        assert status_map["segformer"]["is_champion"] is False
