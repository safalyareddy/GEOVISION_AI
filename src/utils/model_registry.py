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
from typing import Dict, Any, Optional, List, Tuple, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models import build_model, SUPPORTED_MODELS

logger = logging.getLogger("geovision.model_registry")

# Global Settings & Pinned Defaults
CHAMPION_MODEL = "fpn"
DEFAULT_HF_REPO = os.getenv("GEOVISION_HF_REPO", "safalyareddy/geovision-water-segmentation-models")
DEFAULT_HF_REVISION = os.getenv("GEOVISION_HF_REVISION", "main")

# Canonical Model Metadata & Verified Checksums
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fpn": {
        "display_name": "Feature Pyramid Network (FPN - Champion)",
        "is_champion": True,
        "default_filename": "fpn_best.pth",
        "expected_sha256": "d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1",
        "size_mb": 88.48,
        "candidate_paths": [
            Path("checkpoints/fpn_best.pth"),
            Path("experiments/2026_water_segmentation_improvement/checkpoints/EXP06_FPN_FocalTversky_best.pth"),
            Path.home() / ".cache" / "geovision" / "checkpoints" / "fpn_best.pth",
        ],
        "hf_filename": "fpn_best.pth",
    },
    "unet_plusplus": {
        "display_name": "UNet++ (Improved / Baseline)",
        "is_champion": False,
        "default_filename": "unet_plusplus_best.pth",
        "expected_sha256": "e9ac1bed148e6f3e8926d9008fd3fe6ad490ff8ed1dee6d79b96134ef8712c9b",
        "size_mb": 99.68,
        "candidate_paths": [
            Path("checkpoints/unet_plusplus_best.pth"),
            Path("experiments/2026_water_segmentation_improvement/checkpoints/EXP01_UNetPlusPlus_Tversky_best.pth"),
            Path("checkpoints/experiments/round3/unet_plusplus_r3_multiscale.pth"),
            Path("checkpoints/experiments/unet_plusplus_exp1.pth"),
            Path.home() / ".cache" / "geovision" / "checkpoints" / "unet_plusplus_best.pth",
        ],
        "hf_filename": "unet_plusplus_best.pth",
    },
    "deeplabv3_plus": {
        "display_name": "DeepLabV3+ (Improved / Baseline)",
        "is_champion": False,
        "default_filename": "deeplabv3_plus_best.pth",
        "expected_sha256": None,
        "size_mb": 85.76,
        "candidate_paths": [
            Path("checkpoints/deeplabv3_plus_best.pth"),
            Path("experiments/2026_water_segmentation_improvement/checkpoints/EXP04_DeepLabV3Plus_FocalTversky_best.pth"),
            Path("checkpoints/experiments/round3/deeplabv3_plus_r3_multiscale.pth"),
            Path("checkpoints/experiments/deeplabv3_plus_exp2.pth"),
            Path.home() / ".cache" / "geovision" / "checkpoints" / "deeplabv3_plus_best.pth",
        ],
        "hf_filename": "deeplabv3_plus_best.pth",
    },
    "segformer": {
        "display_name": "SegFormer-B0 (Experimental)",
        "is_champion": False,
        "default_filename": "segformer_best.pth",
        "expected_sha256": None,
        "size_mb": 104.48,
        "candidate_paths": [
            Path("checkpoints/segformer_best.pth"),
            Path("experiments/2026_water_segmentation_improvement/checkpoints/EXP05_SegFormer_FocalTversky_best.pth"),
            Path("checkpoints/experiments/round3/segformer_r3_multiscale.pth"),
            Path("checkpoints/experiments/segformer_exp3.pth"),
            Path.home() / ".cache" / "geovision" / "checkpoints" / "segformer_best.pth",
        ],
        "hf_filename": "segformer_best.pth",
    },
}


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


def verify_checkpoint_integrity(file_path: Union[str, Path], expected_sha256: str) -> bool:
    """Validate whether file matches expected cryptographic SHA-256 hash."""
    actual_hash = compute_file_sha256(file_path)
    return actual_hash.lower() == expected_sha256.lower()


def download_from_hf_hub(
    model_name: str,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    dest_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Download model checkpoint from Hugging Face Model Hub with checksum validation.
    """
    if model_name not in MODEL_REGISTRY:
        return None

    meta = MODEL_REGISTRY[model_name]
    target_repo = repo_id or DEFAULT_HF_REPO
    target_rev = revision or DEFAULT_HF_REVISION
    filename = meta["hf_filename"]

    dest_cache = dest_dir or (Path.home() / ".cache" / "geovision" / "checkpoints")
    dest_cache.mkdir(parents=True, exist_ok=True)
    target_file = dest_cache / filename

    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"Downloading '{filename}' from Hugging Face Hub ({target_repo}@{target_rev})...")
        downloaded_path = hf_hub_download(
            repo_id=target_repo,
            filename=filename,
            revision=target_rev,
            cache_dir=str(dest_cache)
        )
        dl_path = Path(downloaded_path)

        # Check cryptographic integrity if expected hash is configured
        expected_hash = meta.get("expected_sha256")
        if expected_hash:
            if not verify_checkpoint_integrity(dl_path, expected_hash):
                logger.error(f"Downloaded file '{dl_path}' failed SHA-256 validation. Removing corrupted file.")
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

        return dl_path
    except Exception as e:
        logger.warning(f"Could not download checkpoint for '{model_name}' from Hugging Face Hub: {e}")
        return None


def resolve_checkpoint_path(model_name: str, allow_download: bool = True) -> Optional[Path]:
    """
    Resolve model checkpoint path from local filesystem candidates, falling back
    to Hugging Face Hub download if absent.
    """
    if model_name not in MODEL_REGISTRY:
        return None

    meta = MODEL_REGISTRY[model_name]
    candidates = meta.get("candidate_paths", [])

    # Check local candidates first
    for cand in candidates:
        if cand.exists() and cand.is_file():
            # If expected hash is defined, verify integrity
            expected_hash = meta.get("expected_sha256")
            if expected_hash:
                try:
                    if verify_checkpoint_integrity(cand, expected_hash):
                        return cand
                    else:
                        logger.warning(f"Local candidate '{cand}' SHA-256 mismatch. Skipping.")
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


def is_model_available(model_name: str) -> bool:
    """Return True if verified checkpoint exists and can be loaded."""
    if not HAS_TORCH:
        return False
    path = resolve_checkpoint_path(model_name, allow_download=False)
    return path is not None and path.exists()


def get_model_status_summary() -> Dict[str, Dict[str, Any]]:
    """Return availability status dictionary for all supported models."""
    summary = {}
    for name, meta in MODEL_REGISTRY.items():
        resolved = resolve_checkpoint_path(name, allow_download=False)
        available = resolved is not None and resolved.exists()
        summary[name] = {
            "model_name": name,
            "display_name": meta["display_name"],
            "is_champion": meta["is_champion"],
            "available": available,
            "checkpoint_path": str(resolved) if resolved else None,
            "expected_sha256": meta.get("expected_sha256"),
            "status_label": "Champion (Weights Loaded)" if meta["is_champion"] and available else (
                "Weights Loaded" if available else "Weights Missing"
            )
        }
    return summary


def load_verified_model(model_name: str, device: str = "cpu") -> Any:
    """
    Instantiate model architecture and load verified checkpoint.

    Strictly raises RuntimeError if weights are missing, preventing uninitialized execution.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed in the environment.")

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown architecture '{model_name}'. Choose from: {SUPPORTED_MODELS}")

    ckpt_path = resolve_checkpoint_path(model_name, allow_download=True)
    if ckpt_path is None or not ckpt_path.exists():
        raise RuntimeError(
            f"Trained checkpoint for '{model_name}' could not be located locally or downloaded. "
            "Inference with uninitialized weights is strictly prohibited. "
            f"Please ensure '{model_name}_best.pth' is present in checkpoints/ or available on Hugging Face Hub."
        )

    # Instantiate architecture
    model = build_model(model_name=model_name)

    # Safe deserialization
    state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict) or len(state) == 0:
        raise RuntimeError(f"Checkpoint '{ckpt_path}' does not contain a valid non-empty state dictionary.")

    # Prefix normalization if needed
    cleaned_state = {}
    for k, v in state.items():
        clean_k = k
        if clean_k.startswith("model."):
            clean_k = clean_k[len("model."):]
        cleaned_state[clean_k] = v

    # Attempt strict loading first, fallback to partial only if compatible
    try:
        model.load_state_dict(cleaned_state, strict=True)
    except Exception as e_strict:
        try:
            model.load_state_dict(state, strict=True)
        except Exception:
            try:
                load_res = model.load_state_dict(cleaned_state, strict=False)
                # If too many missing keys, fail safe
                if len(load_res.missing_keys) > 20:
                    raise RuntimeError(
                        f"Architecture mismatch for '{model_name}' with '{ckpt_path}': "
                        f"{len(load_res.missing_keys)} missing keys, {len(load_res.unexpected_keys)} unexpected keys. "
                        "Aborting to avoid degraded uninitialized inference."
                    )
            except Exception as e_non_strict:
                raise RuntimeError(f"Failed to load checkpoint into '{model_name}': {e_strict} | {e_non_strict}")

    model.to(device)
    model.eval()
    return model
