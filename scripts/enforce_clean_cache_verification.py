"""
GeoVision AI — Clean-Cache Checkpoint Verification Script
Enforces independent SHA-256 calculation and strict PyTorch loading
from an empty temporary cache directly from Hugging Face Hub.
"""

import os
import sys
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Any

from huggingface_hub import hf_hub_download
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import build_model
from src.utils.model_registry import (
    MODEL_REGISTRY,
    AUTHORITATIVE_HF_REPO,
    AUTHORITATIVE_HF_REVISION,
)

EXPECTED_SPEC = {
    "fpn": {
        "filename": "fpn_best.pth",
        "expected_bytes": 92779733,
        "expected_sha256": "d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1",
    },
    "unet_plusplus": {
        "filename": "unet_plusplus_best.pth",
        "expected_bytes": 104528035,
        "expected_sha256": "497e246ef79d87f3b2255cc119e7ab8bd0045d48caf3d4710e9f6738e56c4548",
    },
    "deeplabv3_plus": {
        "filename": "deeplabv3_plus_best.pth",
        "expected_bytes": 89936639,
        "expected_sha256": "6c7c6dd4835d29be34b9eda03bdfca878914749a526c44cade99906cc1b1b827",
    },
    "segformer": {
        "filename": "segformer_best.pth",
        "expected_bytes": 14949211,
        "expected_sha256": "2c762a52a5aa5cafb6aecb698bf483257fd5279bce3f3af7368850ba9354e986",
    },
}


def verify_clean_cache() -> Dict[str, Dict[str, Any]]:
    print("=" * 80)
    print("GeoVision AI — Clean-Cache Checkpoint Verification")
    print(f"Repository: {AUTHORITATIVE_HF_REPO}")
    print(f"Revision:   {AUTHORITATIVE_HF_REVISION}")
    print("=" * 80)

    results = {}
    with tempfile.TemporaryDirectory() as clean_cache_dir:
        print(f"Isolated clean cache directory: {clean_cache_dir}\n")

        for model_key, spec in EXPECTED_SPEC.items():
            fname = spec["filename"]
            print(f"[{model_key.upper()}] Downloading '{fname}' from Hub...")
            dl_path = hf_hub_download(
                repo_id=AUTHORITATIVE_HF_REPO,
                filename=fname,
                revision=AUTHORITATIVE_HF_REVISION,
                cache_dir=clean_cache_dir,
            )
            actual_size = os.path.getsize(dl_path)

            # Compute streaming SHA-256
            hasher = hashlib.sha256()
            with open(dl_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            actual_sha256 = hasher.hexdigest().lower()

            # Verify size and hash match expected
            size_match = actual_size == spec["expected_bytes"]
            sha_match = actual_sha256 == spec["expected_sha256"].lower()

            # Verify it is not a Git LFS pointer text file
            is_not_lfs_pointer = actual_size > 1024

            # Verify strict state_dict loading
            state = torch.load(dl_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            cleaned_state = {k.replace("model.", ""): v for k, v in state.items()}
            model = build_model(model_name=model_key, encoder_weights=None)
            try:
                model.load_state_dict(state, strict=True)
                strict_pass = True
            except Exception:
                try:
                    model.load_state_dict(cleaned_state, strict=True)
                    strict_pass = True
                except Exception as e:
                    strict_pass = False
                    print(f"  Strict loading error: {e}")

            model.eval()
            eval_pass = not model.training

            results[model_key] = {
                "filename": fname,
                "expected_bytes": spec["expected_bytes"],
                "actual_bytes": actual_size,
                "size_match": size_match,
                "expected_sha256": spec["expected_sha256"],
                "actual_sha256": actual_sha256,
                "sha256_match": sha_match,
                "is_not_lfs_pointer": is_not_lfs_pointer,
                "strict_load_pass": strict_pass,
                "eval_mode_pass": eval_pass,
            }

            print(f"  • Size: {actual_size:,} bytes (Expected: {spec['expected_bytes']:,}) -> {'PASS' if size_match else 'FAIL'}")
            print(f"  • SHA-256: {actual_sha256} -> {'PASS' if sha_match else 'FAIL'}")
            print(f"  • LFS Pointer Check: {'PASS (Real Binary)' if is_not_lfs_pointer else 'FAIL (Pointer File)'}")
            print(f"  • Strict PyTorch Loading: {'PASS' if strict_pass else 'FAIL'}")
            print(f"  • Evaluation Mode (model.eval()): {'PASS' if eval_pass else 'FAIL'}\n")

    return results


if __name__ == "__main__":
    res = verify_clean_cache()
    all_ok = all(
        v["size_match"] and v["sha256_match"] and v["is_not_lfs_pointer"] and v["strict_load_pass"] and v["eval_mode_pass"]
        for v in res.values()
    )
    if all_ok:
        print(">>> ALL 4 CHECKPOINTS PASSED CLEAN-CACHE VERIFICATION (100% BITWISE MATCH) <<<")
        sys.exit(0)
    else:
        print(">>> ERROR: Checkpoint verification failed! <<<")
        sys.exit(1)
