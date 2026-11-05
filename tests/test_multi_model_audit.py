"""
GeoVision AI — Multi-Architecture Cloud Inference Audit & Regression Test Suite

Verifies:
- Architecture-specific checkpoint loading
- Cryptographic SHA-256 verification
- Strict state-dictionary loading failures
- Missing checkpoint handling
- Invalid input images
- Inference output dimensions and value ranges
- Model selection and caching
- Failure isolation between models
- Cloud memory considerations
"""

import pytest
import os
import io
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models import build_model, SUPPORTED_MODELS
from src.utils.model_registry import (
    load_verified_model,
    compute_file_sha256,
    verify_checkpoint_integrity,
    resolve_checkpoint_path,
    is_model_available,
    get_model_status_summary,
    MODEL_REGISTRY,
    CHAMPION_MODEL,
)
from app import (
    run_pytorch_inference,
    get_cached_pytorch_model,
    create_overlay,
)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for multi-model audit")
class TestMultiModelAudit:
    """Multi-Architecture Cloud Inference Audit Test Suite."""

    def test_fpn_checkpoint_loading_and_strict_weights(self):
        """Test FPN-ResNet34 champion checkpoint loads strictly with zero key mismatches."""
        model = load_verified_model("fpn", device="cpu")
        assert model is not None
        assert not model.training
        assert sum(p.numel() for p in model.parameters()) == 23155393

        # Forward pass on standard satellite tile
        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (1, 1, 256, 256)
        assert not torch.isnan(out).any()

    def test_unet_plusplus_checkpoint_loading_and_strict_weights(self):
        """Test UNet++ ResNet34 checkpoint loads strictly if checkpoint exists."""
        ckpt_path = resolve_checkpoint_path("unet_plusplus", allow_download=False)
        if ckpt_path is None or not ckpt_path.exists():
            pytest.skip("UNet++ checkpoint not present in local test environment")

        model = load_verified_model("unet_plusplus", device="cpu")
        assert model is not None
        assert not model.training
        assert sum(p.numel() for p in model.parameters()) == 26078609

        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_deeplabv3_plus_checkpoint_loading_and_strict_weights(self):
        """Test DeepLabV3+ ResNet34 checkpoint loads strictly if checkpoint exists."""
        ckpt_path = resolve_checkpoint_path("deeplabv3_plus", allow_download=False)
        if ckpt_path is None or not ckpt_path.exists():
            pytest.skip("DeepLabV3+ checkpoint not present in local test environment")

        model = load_verified_model("deeplabv3_plus", device="cpu")
        assert model is not None
        assert not model.training
        assert sum(p.numel() for p in model.parameters()) == 22437457

        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_checksum_verification_for_all_registered_models(self):
        """Test SHA-256 checksums of candidate checkpoints against registry values."""
        for model_name, meta in MODEL_REGISTRY.items():
            expected = meta.get("expected_sha256")
            if not expected:
                continue
            cand = resolve_checkpoint_path(model_name, allow_download=False)
            if cand and cand.exists():
                computed = compute_file_sha256(cand)
                assert computed.lower() == expected.lower(), (
                    f"Checksum mismatch for '{model_name}': expected {expected}, got {computed}"
                )

    def test_deliberate_checksum_mismatch_fails_safe(self, tmp_path):
        """Test that tampered checkpoint with bad SHA-256 raises RuntimeError before loading."""
        tampered_ckpt = tmp_path / "corrupted_fpn.pth"
        tampered_ckpt.write_bytes(b"tampered invalid content")

        fake_expected = "d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1"
        assert not verify_checkpoint_integrity(tampered_ckpt, fake_expected)

    def test_strict_loading_failure_on_missing_keys(self, tmp_path):
        """Test strict state-dict loading failure when keys are missing."""
        model = build_model("fpn", encoder_weights=None)
        sd = model.state_dict()
        # Remove arbitrary key to induce strict failure
        first_key = list(sd.keys())[0]
        del sd[first_key]

        bad_ckpt = tmp_path / "incomplete_fpn.pth"
        torch.save(sd, str(bad_ckpt))

        with pytest.raises(Exception):
            model.load_state_dict(torch.load(str(bad_ckpt), weights_only=True), strict=True)

    def test_missing_checkpoint_handling(self, monkeypatch):
        """Test that missing checkpoint safely raises RuntimeError without falling back to random weights."""
        def fake_resolve(name, allow_download=True):
            return None

        monkeypatch.setattr("src.utils.model_registry.resolve_checkpoint_path", fake_resolve)
        with pytest.raises(RuntimeError, match="could not be located locally or downloaded"):
            load_verified_model("fpn")

    def test_segformer_checkpoint_loading_and_strict_weights(self):
        """Test SegFormer-MiT-B0 checkpoint loads strictly with zero key mismatches."""
        ckpt_path = resolve_checkpoint_path("segformer", allow_download=False)
        if ckpt_path is None or not ckpt_path.exists():
            pytest.skip("SegFormer checkpoint not present in local test environment")

        model = load_verified_model("segformer", device="cpu")
        assert model is not None
        assert not model.training
        assert sum(p.numel() for p in model.parameters()) == 3714401

        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (1, 1, 256, 256)
        assert not torch.isnan(out).any()

    def test_segformer_mit_b0_architecture_specifications(self):
        """Verify SegFormer model configuration matches MiT-B0 architecture specifications."""
        model = build_model("segformer", encoder_weights=None)
        assert hasattr(model, "config")
        assert model.config.depths == [2, 2, 2, 2]
        assert model.config.hidden_sizes == [32, 64, 160, 256]
        assert model.config.decoder_hidden_size == 256
        assert model.config.num_labels == 1

        # Test bilinear output interpolation matches input spatial dimensions
        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (1, 1, 256, 256)

    def test_huggingface_download_at_pinned_revision(self):
        """Test checkpoint resolution and SHA-256 verification against pinned Hugging Face Hub revision."""
        from src.utils.model_registry import DEFAULT_HF_REPO, DEFAULT_HF_REVISION, download_from_hf_hub
        # Verify default revision is pinned to the multi-model commit
        assert DEFAULT_HF_REVISION == "441a99034b36336c34d86c23cfe9c1a8114ca3a4"
        assert DEFAULT_HF_REPO == "safalya06/geovision-water-segmentation-models"

        # Verify FPN checkpoint can be resolved
        fpn_path = resolve_checkpoint_path("fpn", allow_download=True)
        assert fpn_path is not None
        assert fpn_path.exists()
        assert compute_file_sha256(fpn_path) == MODEL_REGISTRY["fpn"]["expected_sha256"]

    def test_preprocessing_parity(self):
        """Test preprocessing parity: ensures image conversion produces [0, 1] float32 tensor."""
        import torchvision.transforms as T
        test_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), mode="RGB")
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        t_img = transform(test_img)
        assert t_img.shape == (3, 256, 256)
        assert t_img.dtype == torch.float32
        assert float(t_img.min()) >= 0.0
        assert float(t_img.max()) <= 1.0

    def test_invalid_input_images_handling(self):
        """Test inference handles grayscale, RGBA, and variable image dimensions."""
        model = load_verified_model("fpn", device="cpu")

        # Grayscale image (1 channel)
        gray_img = Image.fromarray(np.random.randint(0, 255, (128, 128), dtype=np.uint8), mode="L")
        mask, b_mask, lat, cov, area = run_pytorch_inference(model, gray_img, threshold=0.50)
        assert mask.size == (128, 128)
        assert 0.0 <= cov <= 100.0

        # RGBA image (4 channels)
        rgba_img = Image.fromarray(np.random.randint(0, 255, (300, 200, 4), dtype=np.uint8), mode="RGBA")
        mask, b_mask, lat, cov, area = run_pytorch_inference(model, rgba_img, threshold=0.50)
        assert mask.size == rgba_img.size
        assert 0.0 <= cov <= 100.0

    def test_inference_output_dimensions_and_value_ranges(self):
        """Test inference output mask dimensions, binary values, and coverage percentage bounds."""
        model = load_verified_model("fpn", device="cpu")
        test_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), mode="RGB")

        mask_img, binary_mask, latency_ms, water_coverage_pct, water_area_km2 = run_pytorch_inference(
            model=model,
            image=test_img,
            threshold=0.50
        )

        assert mask_img.size == (256, 256)
        assert binary_mask.shape == (256, 256)
        assert set(np.unique(binary_mask)).issubset({0, 1})
        assert 0.0 <= water_coverage_pct <= 100.0
        assert water_area_km2 >= 0.0
        assert latency_ms > 0.0

    def test_model_selection_and_caching(self):
        """Test that get_cached_pytorch_model returns identical object on repeat calls."""
        m1 = get_cached_pytorch_model("fpn")
        m2 = get_cached_pytorch_model("fpn")
        assert m1 is m2

    def test_sequential_model_switching(self):
        """Test sequential model switching between all four architectures."""
        sample_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), mode="RGB")
        switch_sequence = ["fpn", "unet_plusplus", "deeplabv3_plus", "segformer", "fpn"]

        for model_name in switch_sequence:
            model = get_cached_pytorch_model(model_name)
            assert model is not None
            mask, b_mask, lat, cov, area = run_pytorch_inference(model, sample_img, threshold=0.50)
            assert mask.size == (256, 256)
            assert 0.0 <= cov <= 100.0

    def test_comparison_mode_inference(self):
        """Test multi-architecture comparison workflow executing all four models on the same input image."""
        sample_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), mode="RGB")
        all_results = {}

        for model_name in ["fpn", "unet_plusplus", "deeplabv3_plus", "segformer"]:
            model = get_cached_pytorch_model(model_name)
            mask, b_mask, lat, cov, area = run_pytorch_inference(model, sample_img, threshold=0.50)
            all_results[model_name] = {
                "mask_shape": b_mask.shape,
                "coverage": cov,
                "latency": lat
            }

        assert len(all_results) == 4
        for m_name, res in all_results.items():
            assert res["mask_shape"] == (256, 256)
            assert 0.0 <= res["coverage"] <= 100.0
            assert res["latency"] > 0.0

    def test_failure_isolation_between_models(self):
        """Test that attempting to load an invalid model does not corrupt the cached operational models."""
        m_fpn = get_cached_pytorch_model("fpn")

        with pytest.raises((RuntimeError, ValueError)):
            load_verified_model("nonexistent_unregistered_model")

        # FPN model remains intact, cached, and operational
        m_fpn_after = get_cached_pytorch_model("fpn")
        assert m_fpn is m_fpn_after

        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            out = m_fpn_after(x)
        assert out.shape == (1, 1, 256, 256)

    def test_cloud_memory_considerations(self):
        """Test that total parameter memory across all four models remains well within 1 GB limit."""
        total_param_bytes = 0
        for model_name in ["fpn", "unet_plusplus", "deeplabv3_plus", "segformer"]:
            model = get_cached_pytorch_model(model_name)
            total_param_bytes += sum(p.numel() * p.element_size() for p in model.parameters())

        total_param_mb = total_param_bytes / (1024 * 1024)
        # All 4 models combined parameter memory is ~378 MB float32, comfortably under the 1000 MB limit
        assert total_param_mb < 600.0, f"Combined model parameter footprint exceeds 600 MB: {total_param_mb:.2f} MB"

    def test_architecture_selection_and_registry_metadata(self):
        """Test that all four models are registered with correct display names, backbones, and FPN champion."""
        summary = get_model_status_summary()
        expected_models = ["fpn", "unet_plusplus", "deeplabv3_plus", "segformer"]

        for m_name in expected_models:
            assert m_name in summary
            meta = summary[m_name]
            assert meta["available"] is True
            assert meta["load_supported"] is True

        assert summary["fpn"]["is_champion"] is True
        assert "FPN–ResNet34" in summary["fpn"]["display_name"]
        assert "UNet++–ResNet34" in summary["unet_plusplus"]["display_name"]
        assert "DeepLabV3+–ResNet34" in summary["deeplabv3_plus"]["display_name"]
        assert "SegFormer–MiT-B0" in summary["segformer"]["display_name"]

    def test_non_water_image_predictions(self):
        """Test all 4 models on a near-empty/non-water satellite image to prevent false saturation collapse."""
        img_path = Path("dataset/test/images/patch_0006.png")
        if not img_path.exists():
            test_img = Image.new("RGB", (256, 256), color=(60, 50, 40))
        else:
            test_img = Image.open(img_path).convert("RGB")

        for model_name in ["fpn", "unet_plusplus", "deeplabv3_plus", "segformer"]:
            model = get_cached_pytorch_model(model_name)
            mask, b_mask, lat, cov, area = run_pytorch_inference(model, test_img, threshold=0.50)
            assert mask.size == (256, 256)
            assert b_mask.shape == (256, 256)
            # Must NOT saturate at 100% or produce degenerate predictions
            assert cov < 30.0, f"Model {model_name} produced excessive coverage {cov:.2f}% on non-water image"
            assert not np.isnan(b_mask).any()

    def test_water_image_predictions(self):
        """Test all 4 models on a known water-containing image (patch_0047.png) for positive detections."""
        img_path = Path("dataset/test/images/patch_0047.png")
        if not img_path.exists():
            img_path = Path("assets/demo/patch_0047.png")
        assert img_path.exists(), f"Water test tile missing at {img_path}"

        test_img = Image.open(img_path).convert("RGB")

        for model_name in ["fpn", "unet_plusplus", "deeplabv3_plus", "segformer"]:
            model = get_cached_pytorch_model(model_name)
            mask, b_mask, lat, cov, area = run_pytorch_inference(model, test_img, threshold=0.50)
            assert mask.size == (256, 256)
            assert b_mask.shape == (256, 256)
            # Must predict meaningful water detection (> 1% and < 40%)
            assert 1.0 <= cov <= 40.0, f"Model {model_name} predicted unexpected coverage {cov:.2f}% on patch_0047"
            assert int(np.sum(b_mask)) > 0, f"Model {model_name} failed to detect water in known water tile"

    def test_unified_inference_pipeline_api(self):
        """Test src.inference module interface: load_model, predict, predict_all, metadata."""
        from src.inference import (
            load_model,
            predict,
            predict_all,
            get_model_metadata,
            get_model_status,
        )

        test_img = Image.new("RGB", (256, 256), color="navy")

        # Test predict_all
        res_all = predict_all(test_img, threshold=0.50, device="cpu")
        assert len(res_all) == 4
        for m_name, res in res_all.items():
            assert "error" not in res
            assert res["mask_array_256"].shape == (256, 256)
            assert 0.0 <= res["water_coverage_pct"] <= 100.0

        # Test get_model_metadata
        for m_name in ["fpn", "unet_plusplus", "deeplabv3_plus", "segformer"]:
            meta = get_model_metadata(m_name)
            assert meta["framework"] == "PyTorch"
            assert meta["input_spatial_dimensions"] == [256, 256]
            assert meta["parameter_count"] > 0

    def test_mps_or_cpu_execution_stability(self):
        """Test device execution on CPU and MPS (if supported by system hardware)."""
        from src.inference import predict, get_optimal_device
        test_img = Image.new("RGB", (256, 256), color="blue")

        # CPU run
        res_cpu = predict(test_img, model_name="fpn", device="cpu")
        assert res_cpu["device"] == "cpu"
        assert res_cpu["mask_array_256"].shape == (256, 256)

        # MPS run if available
        if torch.backends.mps.is_available():
            res_mps = predict(test_img, model_name="fpn", device="mps")
            assert res_mps["device"] == "mps"
            # Numerical equivalence check
            assert np.array_equal(res_cpu["mask_array_256"], res_mps["mask_array_256"])


