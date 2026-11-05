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
    download_from_hf_hub,
    is_model_available,
    is_model_obtainable,
    get_model_status_summary,
    load_verified_model,
    get_hf_token,
    CHAMPION_MODEL,
    MODEL_REGISTRY,
    DEFAULT_HF_REPO,
    DEFAULT_HF_REVISION,
    STATUS_VERIFIED_LOADED,
    STATUS_AVAILABLE_NOT_LOADED,
    STATUS_DOWNLOADING,
    STATUS_VERIFICATION_FAILED,
    STATUS_UNAVAILABLE,
    _LAST_DOWNLOAD_ERROR,
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


def test_hf_repo_and_pinned_revision_resolution():
    """Test that default Hugging Face repository and revision are pinned and non-empty."""
    from src.utils.model_registry import DEFAULT_HF_REPO, DEFAULT_HF_REVISION
    assert DEFAULT_HF_REPO == "safalya06/geovision-water-segmentation-models"
    assert DEFAULT_HF_REVISION == "441a99034b36336c34d86c23cfe9c1a8114ca3a4"
    assert len(DEFAULT_HF_REVISION) == 40  # Full git commit sha


def test_deliberately_incorrect_checksum_rejection(tmp_path: Path):
    """Test that a checkpoint with mismatched SHA-256 is strictly rejected."""
    corrupt_file = tmp_path / "corrupt_fpn.pth"
    corrupt_file.write_bytes(b"tampered_or_incomplete_model_weights")
    expected_hash = "d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1"

    # 1. Direct integrity verification must return False
    assert verify_checkpoint_integrity(corrupt_file, expected_hash) is False

    with patch.dict(MODEL_REGISTRY, {"fpn": {
        "display_name": "Tampered FPN",
        "is_champion": True,
        "default_filename": "corrupt_fpn.pth",
        "expected_sha256": expected_hash,
        "candidate_paths": [corrupt_file],
        "hf_filename": "corrupt_fpn.pth"
    }}):
        # 2. Local candidate resolution must refuse to return corrupt file
        rejected_candidate = resolve_checkpoint_path("fpn", allow_download=False)
        assert rejected_candidate is None

        # 3. Model loader must strictly abort rather than loading tampered weights
        with pytest.raises(RuntimeError):
            load_verified_model("fpn", device="cpu")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for strict loading test.")
def test_strict_checkpoint_loading_rejects_missing_keys(tmp_path: Path):
    """Test that state dictionary with missing/incompatible keys fails strictly without random fallback."""
    bad_dict_file = tmp_path / "bad_dict.pth"
    torch.save({"partial_weight": torch.randn(1, 1, 3, 3)}, str(bad_dict_file))

    with patch.dict(MODEL_REGISTRY, {"fpn": {
        "display_name": "Incompatible Dict FPN",
        "is_champion": True,
        "default_filename": "bad_dict.pth",
        "expected_sha256": None,
        "candidate_paths": [bad_dict_file],
        "hf_filename": "bad_dict.pth"
    }}):
        with pytest.raises(RuntimeError) as excinfo:
            load_verified_model("fpn", device="cpu")
        assert "Strict state-dictionary loading failed" in str(excinfo.value)


def test_get_hf_token_from_env_and_secrets(monkeypatch):
    """Test optional HF token resolution from environment variables and streamlit secrets."""
    # 1. When no token is set
    for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert get_hf_token() is None

    # 2. When env var is set
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_12345")
    assert get_hf_token() == "hf_test_token_12345"

    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "hf_hub_token_abc")
    assert get_hf_token() == "hf_hub_token_abc"


def test_verify_checkpoint_integrity_multi_hash(tmp_path: Path):
    """Test cryptographic integrity check against both single strings and lists of hashes."""
    sample_file = tmp_path / "model.pth"
    sample_file.write_bytes(b"test_checkpoint_content_2026")
    actual_hash = compute_file_sha256(sample_file)

    # Matches single string
    assert verify_checkpoint_integrity(sample_file, actual_hash) is True
    assert verify_checkpoint_integrity(sample_file, actual_hash.upper()) is True

    # Matches list of strings containing the correct hash
    trusted_list = ["fake_hash_1", actual_hash, "fake_hash_2"]
    assert verify_checkpoint_integrity(sample_file, trusted_list) is True

    # Fails when list contains only invalid hashes
    assert verify_checkpoint_integrity(sample_file, ["fake_hash_1", "fake_hash_2"]) is False


def test_hf_download_empty_file_rejected(tmp_path: Path):
    """Test that an empty (0 byte) file downloaded from HF is rejected and unlinked."""
    empty_file = tmp_path / "empty_model.pth"
    empty_file.touch()

    with patch("huggingface_hub.hf_hub_download", return_value=str(empty_file)):
        with patch.dict(MODEL_REGISTRY, {"test_empty": {
            "display_name": "Empty Model",
            "is_champion": False,
            "default_filename": "empty_model.pth",
            "expected_sha256": "any_hash",
            "hf_filename": "empty_model.pth",
            "is_hosted_on_hf": True,
        }}):
            dl = download_from_hf_hub("test_empty", dest_dir=tmp_path)
            assert dl is None
            assert not empty_file.exists()  # Corrupted 0-byte file must be deleted
            assert "empty or invalid (0 bytes)" in _LAST_DOWNLOAD_ERROR["test_empty"]


def test_hf_download_hash_mismatch_rejected(tmp_path: Path):
    """Test that downloaded file with wrong hash is unlinked and recorded in diagnostics."""
    corrupt_file = tmp_path / "corrupt_hf.pth"
    corrupt_file.write_bytes(b"corrupt_download_bytes")

    with patch("huggingface_hub.hf_hub_download", return_value=str(corrupt_file)):
        with patch.dict(MODEL_REGISTRY, {"test_corrupt": {
            "display_name": "Corrupt Model",
            "is_champion": False,
            "default_filename": "corrupt_hf.pth",
            "expected_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "trusted_sha256_list": ["0000000000000000000000000000000000000000000000000000000000000000"],
            "hf_filename": "corrupt_hf.pth",
            "is_hosted_on_hf": True,
        }}):
            dl = download_from_hf_hub("test_corrupt", dest_dir=tmp_path)
            assert dl is None
            assert not corrupt_file.exists()  # Must be cleaned up
            assert "failed cryptographic SHA-256 validation" in _LAST_DOWNLOAD_ERROR["test_corrupt"]


def test_hf_download_failure_captures_diagnostic_and_surfaces_in_load(tmp_path: Path):
    """Test that underlying HF exception is captured in error message and not silently swallowed."""
    with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("EntryNotFoundError: 404 Client Error")):
        with patch.dict(MODEL_REGISTRY, {"fpn": {
            "display_name": "Champion FPN",
            "is_champion": True,
            "default_filename": "fpn_404.pth",
            "expected_sha256": "some_hash",
            "candidate_paths": [tmp_path / "nonexistent_404.pth"],
            "hf_filename": "fpn_404.pth",
            "is_hosted_on_hf": True,
            "load_supported": True,
        }}):
            with pytest.raises(RuntimeError) as excinfo:
                load_verified_model("fpn", device="cpu")
            err_text = str(excinfo.value)
            assert "EntryNotFoundError: 404 Client Error" in err_text
            assert "safalya06/geovision-water-segmentation-models" in err_text
            assert "441a99034b36336c34d86c23cfe9c1a8114ca3a4" in err_text


def test_model_status_categories():
    """Test that model status summary provides canonical 5 status labels."""
    summary = get_model_status_summary()
    valid_statuses = {
        STATUS_VERIFIED_LOADED,
        STATUS_AVAILABLE_NOT_LOADED,
        STATUS_DOWNLOADING,
        STATUS_VERIFICATION_FAILED,
        STATUS_UNAVAILABLE,
    }
    for model_name, info in summary.items():
        assert info["status_label"] in valid_statuses, f"Invalid status: {info['status_label']}"

