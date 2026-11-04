"""
GeoVision AI Satellite Water Body Segmentation Platform
Unit & Integration Tests for Model Registry and Cryptographic Verification
"""

import io
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.utils.model_registry import (
    compute_file_sha256,
    verify_checkpoint_integrity,
    resolve_checkpoint_path,
    is_model_available,
    get_model_status_summary,
    load_verified_model,
    CHAMPION_MODEL,
    MODEL_REGISTRY,
)


def test_compute_file_sha256_known_content(tmp_path: Path):
    """Test SHA-256 hash calculation against known string."""
    test_file = tmp_path / "sample.bin"
    test_file.write_bytes(b"GeoVisionAI_2026_Water_Segmentation")
    # hashlib.sha256(b"GeoVisionAI_2026_Water_Segmentation").hexdigest()
    expected_hash = "4440acbaf8be3331804ebf8ed862c35917789fd06d365937652da3646f0b86a7"
    computed = compute_file_sha256(test_file)
    assert computed == expected_hash


def test_compute_file_sha256_missing_file():
    """Test that nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        compute_file_sha256(Path("nonexistent_path/fake_file.pth"))


def test_verify_checkpoint_integrity(tmp_path: Path):
    """Test integrity verification matching and mismatch."""
    test_file = tmp_path / "model.pth"
    test_file.write_bytes(b"dummy_weights")
    correct_hash = compute_file_sha256(test_file)

    assert verify_checkpoint_integrity(test_file, correct_hash) is True
    assert verify_checkpoint_integrity(test_file, "0000000000000000000000000000000000000000000000000000000000000000") is False


def test_model_status_summary():
    """Test that model status summary provides metadata for all models."""
    summary = get_model_status_summary()
    assert CHAMPION_MODEL in summary
    assert summary[CHAMPION_MODEL]["is_champion"] is True
    assert "display_name" in summary[CHAMPION_MODEL]
    assert "status_label" in summary[CHAMPION_MODEL]


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for model loading tests.")
def test_resolve_and_load_champion_fpn():
    """Test that champion FPN checkpoint is located, verified, and loaded."""
    path = resolve_checkpoint_path(CHAMPION_MODEL, allow_download=False)
    if path is not None and path.exists():
        expected_hash = MODEL_REGISTRY[CHAMPION_MODEL]["expected_sha256"]
        assert verify_checkpoint_integrity(path, expected_hash) is True

        model = load_verified_model(CHAMPION_MODEL, device="cpu")
        assert model is not None
        assert not model.training  # Must be in eval mode


def test_load_verified_model_invalid_architecture():
    """Test that requesting an invalid architecture raises ValueError."""
    with pytest.raises(ValueError):
        load_verified_model("fictional_architecture_v99")


def test_load_verified_model_missing_checkpoint_fails_safe(tmp_path: Path):
    """Test that missing checkpoint strictly prevents uninitialized inference."""
    with patch.dict(MODEL_REGISTRY, {"fpn": {
        "display_name": "Test FPN",
        "is_champion": True,
        "default_filename": "nonexistent.pth",
        "expected_sha256": None,
        "candidate_paths": [tmp_path / "nonexistent.pth"],
        "hf_filename": "nonexistent.pth"
    }}):
        with pytest.raises(RuntimeError) as excinfo:
            load_verified_model("fpn", device="cpu")
        assert "uninitialized weights is strictly prohibited" in str(excinfo.value)
