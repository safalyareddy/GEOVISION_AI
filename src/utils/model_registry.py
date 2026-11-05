"""
GeoVision AI — Model Artifact Registry & Checkpoint Verification Engine

Provides cryptographic verification, local path resolution, automated Hugging Face
Hub downloading, safe model loading, and availability checks to guarantee zero
uninitialized or simulated inferences.
"""

import os
import io
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Set

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models import build_model, SUPPORTED_MODELS

logger = logging.getLogger("geovision.model_registry")

# Global Settings & Pinned Defaults
CHAMPION_MODEL = "fpn"
DEFAULT_HF_REPO = os.getenv("GEOVISION_HF_REPO", "safalya06/geovision-water-segmentation-models")
DEFAULT_HF_REVISION = os.getenv("GEOVISION_HF_REVISION", "441a99034b36336c34d86c23cfe9c1a8114ca3a4")

# Global In-Memory Tracking for Runtime Diagnostics & Caching
_LAST_DOWNLOAD_ERROR: Dict[str, Optional[str]] = {}
_DOWNLOADING_MODELS: Set[str] = set()
_LOADED_MODELS_CACHE: Dict[str, Any] = {}

# Canonical UI Model Status Constants
STATUS_VERIFIED_LOADED = "Verified and loaded"
STATUS_AVAILABLE_NOT_LOADED = "Available but not yet loaded"
STATUS_DOWNLOADING = "Downloading"
STATUS_VERIFICATION_FAILED = "Checkpoint verification failed"
STATUS_UNAVAILABLE = "Unavailable"

# Canonical Model Metadata & Verified Checksums
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fpn": {
        "display_name": "FPN–ResNet34 (Champion Model)",
        "is_champion": True,
        "default_filename": "fpn_best.pth",
        "expected_sha256": "d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1",
        "trusted_sha256_list": [
            "d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1",
        ],
        "size_mb": 88.48,
        "candidate_paths": [
            Path("checkpoints/fpn_best.pth"),
            Path("experiments/2026_water_segmentation_improvement/checkpoints/EXP06_FPN_FocalTversky_best.pth"),
            Path.home() / ".cache" / "geovision" / "checkpoints" / "fpn_best.pth",
        ],
        "hf_filename": "fpn_best.pth",
        "is_hosted_on_hf": True,
        "load_supported": True,
        "support_note": "Authoritative EXP06 champion checkpoint, hosted on Hugging Face Hub, verified for cloud inference.",
    },
    "unet_plusplus": {
        "display_name": "UNet++–ResNet34 (Improved Model)",
        "is_champion": False,
        "default_filename": "unet_plusplus_best.pth",
        "expected_sha256": "497e246ef79d87f3b2255cc119e7ab8bd0045d48caf3d4710e9f6738e56c4548",
        "trusted_sha256_list": [
            "497e246ef79d87f3b2255cc119e7ab8bd0045d48caf3d4710e9f6738e56c4548",  # Authoritative EXP01 checkpoint
            "e9ac1bed148e6f3e8926d9008fd3fe6ad490ff8ed1dee6d79b96134ef8712c9b",  # Legacy baseline
        ],
        "size_mb": 99.69,
        "candidate_paths": [
            Path("checkpoints/unet_plusplus_best.pth"),
            Path("experiments/2026_water_segmentation_improvement/checkpoints/EXP01_UNetPlusPlus_Tversky_best.pth"),
            Path("checkpoints/experiments/unet_plusplus_baseline_legacy_e9ac.pth"),
            Path.home() / ".cache" / "geovision" / "checkpoints" / "unet_plusplus_best.pth",
        ],
        "hf_filename": "unet_plusplus_best.pth",
        "is_hosted_on_hf": True,
        "load_supported": True,
        "support_note": "Authoritative EXP01 Tversky checkpoint, hosted on Hugging Face Hub, verified with 100% strict PyTorch loading.",
    },
    "deeplabv3_plus": {
        "display_name": "DeepLabV3+–ResNet34 (Improved Model)",
        "is_champion": False,
        "default_filename": "deeplabv3_plus_best.pth",
        "expected_sha256": "6c7c6dd4835d29be34b9eda03bdfca878914749a526c44cade99906cc1b1b827",
        "trusted_sha256_list": [
            "6c7c6dd4835d29be34b9eda03bdfca878914749a526c44cade99906cc1b1b827",  # Authoritative EXP04 checkpoint
            "58efb3a365479b99483bfc014c301050306456747396262ef1bec3ea7646201b",  # Legacy degenerate
        ],
        "size_mb": 85.77,
        "candidate_paths": [
            Path("checkpoints/deeplabv3_plus_best.pth"),
            Path("experiments/2026_water_segmentation_improvement/checkpoints/EXP04_DeepLabV3Plus_FocalTversky_best.pth"),
            Path("checkpoints/experiments/degenerate_deeplabv3_plus_58ef.pth"),
            Path.home() / ".cache" / "geovision" / "checkpoints" / "deeplabv3_plus_best.pth",
        ],
        "hf_filename": "deeplabv3_plus_best.pth",
        "is_hosted_on_hf": True,
        "load_supported": True,
        "support_note": "Authoritative EXP04 Focal Tversky checkpoint, hosted on Hugging Face Hub, verified with 100% strict PyTorch loading.",
    },
    "segformer": {
        "display_name": "SegFormer–MiT-B0 (Improved Model)",
        "is_champion": False,
        "default_filename": "segformer_best.pth",
        "expected_sha256": "2c762a52a5aa5cafb6aecb698bf483257fd5279bce3f3af7368850ba9354e986",
        "trusted_sha256_list": [
            "2c762a52a5aa5cafb6aecb698bf483257fd5279bce3f3af7368850ba9354e986",  # Authoritative EXP05 checkpoint
        ],
        "size_mb": 14.26,
        "candidate_paths": [
            Path("checkpoints/segformer_best.pth"),
            Path("experiments/2026_water_segmentation_improvement/checkpoints/EXP05_SegFormer_FocalTversky_best.pth"),
            Path.home() / ".cache" / "geovision" / "checkpoints" / "segformer_best.pth",
        ],
        "hf_filename": "segformer_best.pth",
        "is_hosted_on_hf": True,
        "load_supported": True,
        "support_note": "Authoritative EXP05 Focal Tversky checkpoint, hosted on Hugging Face Hub, verified with 100% strict PyTorch loading.",
    },
}


def get_hf_token() -> Optional[str]:
    """
    Retrieve optional Hugging Face Hub authentication token from environment
    variables or Streamlit secrets (for private/gated repositories or rate limits).
    """
    # 1. Environment variables
    for env_key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN"):
        val = os.getenv(env_key)
        if val and val.strip():
            return val.strip()

    # 2. Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for secret_key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN"):
                if secret_key in st.secrets:
                    secret_val = st.secrets[secret_key]
                    if secret_val and str(secret_val).strip():
                        return str(secret_val).strip()
    except Exception:
        pass

    return None


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 checksum of a file in 64 KB chunks."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot compute checksum: '{path}' is not a valid file.")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checkpoint_integrity(
    file_path: Union[str, Path],
    expected_sha256: Union[str, List[str], Tuple[str, ...], Set[str]]
) -> bool:
    """Validate whether file matches expected cryptographic SHA-256 hash or list of trusted hashes."""
    actual_hash = compute_file_sha256(file_path).lower()
    if isinstance(expected_sha256, str):
        return actual_hash == expected_sha256.lower()
    trusted = [h.lower() for h in expected_sha256 if h]
    return actual_hash in trusted


def download_from_hf_hub(
    model_name: str,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    dest_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Download model checkpoint from Hugging Face Model Hub with checksum validation
    and explicit diagnostic error capture.
    """
    if model_name not in MODEL_REGISTRY:
        _LAST_DOWNLOAD_ERROR[model_name] = f"Architecture '{model_name}' is not in MODEL_REGISTRY."
        return None

    meta = MODEL_REGISTRY[model_name]
    if not repo_id and not meta.get("is_hosted_on_hf", False):
        err = (
            f"Model '{model_name}' is not hosted on Hugging Face Hub ({DEFAULT_HF_REPO}). "
            "Checkpoint is available for local and Docker container environments only."
        )
        logger.info(err)
        _LAST_DOWNLOAD_ERROR[model_name] = err
        return None

    target_repo = repo_id or DEFAULT_HF_REPO
    target_rev = revision or DEFAULT_HF_REVISION
    filename = meta["hf_filename"]

    dest_cache = dest_dir or (Path.home() / ".cache" / "geovision" / "checkpoints")
    dest_cache.mkdir(parents=True, exist_ok=True)
    target_file = dest_cache / filename

    _DOWNLOADING_MODELS.add(model_name)
    try:
        from huggingface_hub import hf_hub_download
        token = get_hf_token()
        logger.info(f"Downloading '{filename}' from Hugging Face Hub ({target_repo}@{target_rev})...")
        downloaded_path = hf_hub_download(
            repo_id=target_repo,
            filename=filename,
            revision=target_rev,
            cache_dir=str(dest_cache),
            token=token
        )
        dl_path = Path(downloaded_path)

        # Verify file exists and is non-empty
        if not dl_path.is_file() or dl_path.stat().st_size == 0:
            err = f"Downloaded file '{dl_path}' is empty or invalid (0 bytes)."
            logger.error(err)
            _LAST_DOWNLOAD_ERROR[model_name] = err
            if dl_path.exists():
                dl_path.unlink()
            return None

        # Check cryptographic integrity if trusted hashes are configured
        trusted_hashes = meta.get("trusted_sha256_list") or ([meta["expected_sha256"]] if meta.get("expected_sha256") else [])
        if trusted_hashes:
            if not verify_checkpoint_integrity(dl_path, trusted_hashes):
                actual_hash = compute_file_sha256(dl_path)
                err = (
                    f"Downloaded checkpoint '{filename}' failed cryptographic SHA-256 validation. "
                    f"Calculated: '{actual_hash}', trusted: {trusted_hashes}. Removing corrupted file."
                )
                logger.error(err)
                _LAST_DOWNLOAD_ERROR[model_name] = err
                if dl_path.exists():
                    dl_path.unlink()
                return None

        # Link or copy to target destination if distinct
        if dl_path != target_file and not target_file.exists():
            try:
                target_file.symlink_to(dl_path)
            except Exception:
                import shutil
                shutil.copy2(dl_path, target_file)

        _LAST_DOWNLOAD_ERROR[model_name] = None
        return dl_path
    except Exception as e:
        err = f"Hugging Face download exception ({type(e).__name__}): {e}"
        logger.warning(f"Could not download checkpoint for '{model_name}' from Hugging Face Hub: {err}")
        _LAST_DOWNLOAD_ERROR[model_name] = err
        return None
    finally:
        _DOWNLOADING_MODELS.discard(model_name)


def resolve_checkpoint_path(model_name: str, allow_download: bool = True) -> Optional[Path]:
    """
    Resolve model checkpoint path from local filesystem candidates, falling back
    to Hugging Face Hub download if absent.
    """
    if model_name not in MODEL_REGISTRY:
        return None

    meta = MODEL_REGISTRY[model_name]
    candidates = meta.get("candidate_paths", [])
    trusted_hashes = meta.get("trusted_sha256_list") or ([meta["expected_sha256"]] if meta.get("expected_sha256") else [])

    # Check local candidates first
    for cand in candidates:
        if cand.exists() and cand.is_file():
            if trusted_hashes:
                try:
                    if verify_checkpoint_integrity(cand, trusted_hashes):
                        return cand
                    else:
                        logger.warning(f"Local candidate '{cand}' SHA-256 mismatch against trusted hashes. Skipping.")
                        continue
                except Exception as e:
                    logger.warning(f"Error checking hash for '{cand}': {e}")
                    continue
            return cand

    # If missing locally and auto-download enabled, try Hugging Face Hub
    if allow_download:
        dl_path = download_from_hf_hub(model_name)
        if dl_path and dl_path.exists():
            return dl_path

    return None


def is_model_available(model_name: str, allow_download: bool = False) -> bool:
    """Return True if verified checkpoint exists, can be loaded, and is supported."""
    if not HAS_TORCH:
        return False
    if model_name not in MODEL_REGISTRY:
        return False
    meta = MODEL_REGISTRY[model_name]
    if not meta.get("load_supported", True):
        return False
    path = resolve_checkpoint_path(model_name, allow_download=allow_download)
    return path is not None and path.exists()


def is_model_obtainable(model_name: str) -> bool:
    """Return True if model checkpoint is available locally or configured on Hugging Face Hub."""
    if model_name not in MODEL_REGISTRY:
        return False
    meta = MODEL_REGISTRY[model_name]
    if not meta.get("load_supported", True):
        return False
    path = resolve_checkpoint_path(model_name, allow_download=False)
    if path and path.exists():
        return True
    return bool(meta.get("is_hosted_on_hf") and meta.get("hf_filename") and DEFAULT_HF_REPO)


def get_model_status_summary() -> Dict[str, Dict[str, Any]]:
    """
    Return comprehensive availability and verification status dictionary for all supported models.
    Supports 5 standard status categories:
    - Verified and loaded
    - Available but not yet loaded
    - Downloading
    - Checkpoint verification failed
    - Unavailable
    """
    summary = {}
    for name, meta in MODEL_REGISTRY.items():
        trusted_hashes = meta.get("trusted_sha256_list") or ([meta["expected_sha256"]] if meta.get("expected_sha256") else [])
        is_supported = meta.get("load_supported", True)
        is_hosted = meta.get("is_hosted_on_hf", False)

        # Check local candidate paths
        resolved = resolve_checkpoint_path(name, allow_download=False)
        has_file = resolved is not None and resolved.exists()

        # Check if candidate exists on disk but failed verification
        has_corrupted_candidate = False
        if not has_file:
            for cand in meta.get("candidate_paths", []):
                if cand.exists() and cand.is_file():
                    try:
                        if not verify_checkpoint_integrity(cand, trusted_hashes):
                            has_corrupted_candidate = True
                            break
                    except Exception:
                        has_corrupted_candidate = True
                        break

        # Check if loaded in memory
        is_loaded = name in _LOADED_MODELS_CACHE
        is_downloading = name in _DOWNLOADING_MODELS

        if is_loaded:
            status_label = STATUS_VERIFIED_LOADED
        elif is_downloading:
            status_label = STATUS_DOWNLOADING
        elif has_file and is_supported:
            status_label = STATUS_AVAILABLE_NOT_LOADED
        elif has_corrupted_candidate:
            status_label = STATUS_VERIFICATION_FAILED
        elif is_hosted:
            # On cloud deployments, hosted models are obtainable and ready to load
            status_label = STATUS_AVAILABLE_NOT_LOADED
        else:
            status_label = STATUS_UNAVAILABLE

        summary[name] = {
            "model_name": name,
            "display_name": meta["display_name"],
            "is_champion": meta["is_champion"],
            "available": (has_file and is_supported) or is_hosted,
            "is_loaded": is_loaded,
            "is_hosted_on_hf": is_hosted,
            "load_supported": is_supported,
            "support_note": meta.get("support_note", ""),
            "checkpoint_path": str(resolved) if resolved else None,
            "expected_sha256": meta.get("expected_sha256"),
            "trusted_sha256_list": trusted_hashes,
            "pinned_revision": DEFAULT_HF_REVISION if is_hosted else None,
            "hf_repo": DEFAULT_HF_REPO if is_hosted else None,
            "status_label": status_label,
            "last_error": _LAST_DOWNLOAD_ERROR.get(name),
        }
    return summary


def load_verified_model(model_name: str, device: str = "cpu") -> Any:
    """
    Instantiate model architecture and load verified checkpoint.

    Strictly raises RuntimeError if weights are missing, compromised, or mismatched,
    preventing uninitialized or simulated execution.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed in the environment.")

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown architecture '{model_name}'. Choose from: {SUPPORTED_MODELS}")

    if not MODEL_REGISTRY[model_name].get("load_supported", True):
        note = MODEL_REGISTRY[model_name].get("support_note", "Architecture unsupported")
        raise RuntimeError(
            f"Architecture '{model_name}' cannot be loaded: {note}. "
            "Inference with uninitialized or incompatible weights is strictly prohibited."
        )

    ckpt_path = resolve_checkpoint_path(model_name, allow_download=True)
    if ckpt_path is None or not ckpt_path.exists():
        diag_err = _LAST_DOWNLOAD_ERROR.get(model_name)
        diag_details = f"\nUnderlying Hugging Face diagnostic: {diag_err}" if diag_err else ""
        raise RuntimeError(
            f"Trained checkpoint for '{model_name}' could not be located locally or downloaded from Hugging Face Hub.\n"
            f"Expected checkpoint: {MODEL_REGISTRY[model_name].get('default_filename')}\n"
            f"Configured Hugging Face repository: {DEFAULT_HF_REPO}\n"
            f"Pinned revision: {DEFAULT_HF_REVISION}{diag_details}\n"
            "Inference with uninitialized weights is strictly prohibited."
        )

    # 1. Cryptographic SHA-256 Verification prior to deserialization
    trusted_hashes = MODEL_REGISTRY[model_name].get("trusted_sha256_list") or (
        [MODEL_REGISTRY[model_name]["expected_sha256"]] if MODEL_REGISTRY[model_name].get("expected_sha256") else []
    )
    if trusted_hashes:
        if not verify_checkpoint_integrity(ckpt_path, trusted_hashes):
            actual_hash = compute_file_sha256(ckpt_path)
            raise RuntimeError(
                f"Cryptographic SHA-256 verification failed for '{ckpt_path}'. "
                f"Trusted hashes: {trusted_hashes}, calculated: {actual_hash}. "
                "Inference with compromised or mismatched weights is strictly prohibited."
            )

    # 2. Instantiate exact architecture with no uninitialized/random ImageNet download
    model = build_model(model_name=model_name, encoder_weights=None)

    # 3. Safe deserialization using PyTorch weights_only=True
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict) or len(state) == 0:
        raise RuntimeError(f"Checkpoint '{ckpt_path}' does not contain a valid non-empty state dictionary.")

    # 4. Prefix normalization (e.g. PyTorch Lightning 'model.' prefix)
    cleaned_state = {}
    for k, v in state.items():
        clean_k = k
        if clean_k.startswith("model."):
            clean_k = clean_k[len("model."):]
        cleaned_state[clean_k] = v

    # 5. Strict state-dictionary loading
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        try:
            model.load_state_dict(cleaned_state, strict=True)
        except Exception as e_strict:
            raise RuntimeError(
                f"Strict state-dictionary loading failed for '{model_name}' using checkpoint '{ckpt_path}': {e_strict}. "
                "Inference with mismatched or partially initialized weights is strictly prohibited."
            )

    # 6. Verify evaluation mode and device placement
    model.to(device)
    model.eval()

    # Safety assertion: guarantee evaluation mode
    if model.training:
        raise RuntimeError(f"Safety violation: Model '{model_name}' is in training mode after loading. Must be in eval mode.")

    _LOADED_MODELS_CACHE[model_name] = model
    return model
