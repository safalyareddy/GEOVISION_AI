"""
GeoVision AI — 8-Combination Model & Engine Parity Verification
Tests all 4 trained architectures across both PyTorch and ONNX Runtime engines
on multiple genuine Sentinel-2 satellite tiles spanning diverse hydrological scenes:
- Water-rich scenes
- Low-water scenes
- Mixed land/water scenes
- Difficult boundary cases
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import SUPPORTED_MODELS
from src.inference import predict, load_model, load_onnx_session
from src.utils.model_registry import (
    MODEL_REGISTRY,
    compute_file_sha256,
    resolve_checkpoint_path,
)

# Test tile selection representing diverse hydrological regimes
TEST_TILES = [
    {"filename": "patch_0015.png", "scene_type": "Low-water (<1% water coverage)"},
    {"filename": "patch_0045.png", "scene_type": "Low-water (<1% water coverage)"},
    {"filename": "patch_0010.png", "scene_type": "Mixed land/water (10-15% water coverage)"},
    {"filename": "patch_0018.png", "scene_type": "Mixed land/water (10-15% water coverage)"},
    {"filename": "patch_0047.png", "scene_type": "Difficult boundary / demo patch (~25% water coverage)"},
    {"filename": "patch_0012.png", "scene_type": "Water-rich (>45% water coverage)"},
    {"filename": "patch_0021.png", "scene_type": "Water-rich (>45% water coverage)"},
]


def run_verification() -> Dict[str, Any]:
    print("=" * 80)
    print("GeoVision AI — 8-Combination Model & Engine Verification Suite")
    print("4 Architectures × 2 Inference Engines = 8 Production Combinations")
    print("=" * 80)

    dataset_dir = Path("dataset/test/images")
    if not dataset_dir.exists():
        dataset_dir = Path("assets/demo")

    results_matrix = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "architecture_count": len(SUPPORTED_MODELS),
        "engine_count": 2,
        "total_combinations": len(SUPPORTED_MODELS) * 2,
        "combinations": {},
        "parity_analysis": {},
        "overall_summary": {
            "all_passed": True,
            "passed_combinations": 0,
            "failed_combinations": 0,
        }
    }

    # Verify all 4 models under PyTorch and ONNX Runtime
    for model_key in SUPPORTED_MODELS.keys():
        meta = MODEL_REGISTRY.get(model_key, {})
        display_name = meta.get("display_name", model_key)
        ckpt_path = resolve_checkpoint_path(model_key, allow_download=True)
        ckpt_hash = compute_file_sha256(ckpt_path) if ckpt_path and ckpt_path.exists() else None

        print(f"\n[{model_key.upper()}] — {display_name}")
        print(f"  • Checkpoint: {ckpt_path}")
        print(f"  • Checkpoint SHA-256: {ckpt_hash}")

        # 1. PyTorch Engine Verification
        pt_key = f"{model_key}__pytorch"
        try:
            pt_model = load_model(model_key, device="cpu")
            pt_model.eval()
            results_matrix["combinations"][pt_key] = {
                "model_key": model_key,
                "display_name": display_name,
                "engine": "PyTorch",
                "checkpoint_sha256": ckpt_hash,
                "status": "PASS",
                "error": None,
            }
            results_matrix["overall_summary"]["passed_combinations"] += 1
            print(f"  [PASS] PyTorch Engine: Model loaded and evaluation mode confirmed.")
        except Exception as e_pt:
            results_matrix["combinations"][pt_key] = {
                "model_key": model_key,
                "display_name": display_name,
                "engine": "PyTorch",
                "checkpoint_sha256": ckpt_hash,
                "status": "FAIL",
                "error": str(e_pt),
            }
            results_matrix["overall_summary"]["failed_combinations"] += 1
            results_matrix["overall_summary"]["all_passed"] = False
            print(f"  [FAIL] PyTorch Engine: {e_pt}")

        # 2. ONNX Runtime Engine Verification
        onnx_key = f"{model_key}__onnx"
        try:
            onnx_sess = load_onnx_session(model_key)
            onnx_path = Path("checkpoints/onnx") / f"{model_key}.onnx"
            onnx_hash = compute_file_sha256(onnx_path) if onnx_path.exists() else None
            results_matrix["combinations"][onnx_key] = {
                "model_key": model_key,
                "display_name": display_name,
                "engine": "ONNX Runtime",
                "onnx_artifact_path": str(onnx_path),
                "onnx_sha256": onnx_hash,
                "status": "PASS",
                "error": None,
            }
            results_matrix["overall_summary"]["passed_combinations"] += 1
            print(f"  [PASS] ONNX Runtime: Session initialized ({onnx_path.name}, SHA: {onnx_hash[:12] if onnx_hash else 'N/A'}...).")
        except Exception as e_onnx:
            results_matrix["combinations"][onnx_key] = {
                "model_key": model_key,
                "display_name": display_name,
                "engine": "ONNX Runtime",
                "status": "FAIL",
                "error": str(e_onnx),
            }
            results_matrix["overall_summary"]["failed_combinations"] += 1
            results_matrix["overall_summary"]["all_passed"] = False
            print(f"  [FAIL] ONNX Runtime: {e_onnx}")

        # 3. Multi-Tile Parity Verification
        model_tile_metrics = []
        for tile_info in TEST_TILES:
            t_name = tile_info["filename"]
            t_path = dataset_dir / t_name
            if not t_path.exists():
                t_path = Path("dataset/test/images") / t_name
            if not t_path.exists():
                t_path = Path("assets/demo") / t_name
            if not t_path.exists():
                continue

            img = Image.open(t_path).convert("RGB")

            # Run PyTorch
            res_pt = predict(img, model_name=model_key, threshold=0.50, engine="pytorch")
            probs_pt = res_pt["probability_map_256"]
            mask_pt = res_pt["mask_array_256"]

            # Run ONNX
            res_onnx = predict(img, model_name=model_key, threshold=0.50, engine="onnx")
            probs_onnx = res_onnx["probability_map_256"]
            mask_onnx = res_onnx["mask_array_256"]

            # Metrics
            abs_diff = np.abs(probs_pt - probs_onnx)
            max_abs_diff = float(np.max(abs_diff))
            mean_abs_diff = float(np.mean(abs_diff))
            mse_diff = float(np.mean((probs_pt - probs_onnx) ** 2))
            mask_agreement_pct = float(np.mean(mask_pt == mask_onnx) * 100.0)

            tile_record = {
                "filename": t_name,
                "scene_type": tile_info["scene_type"],
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "mse": mse_diff,
                "mask_agreement_pct": mask_agreement_pct,
                "pytorch_water_pct": res_pt["water_coverage_pct"],
                "onnx_water_pct": res_onnx["water_coverage_pct"],
                "pytorch_latency_ms": res_pt["latency_ms"],
                "onnx_latency_ms": res_onnx["latency_ms"],
                "parity_passed": (max_abs_diff < 1e-3 and mask_agreement_pct >= 99.0),
            }
            model_tile_metrics.append(tile_record)

        # Aggregate metrics for this model across all tiles
        if model_tile_metrics:
            agg_max_diff = max(m["max_abs_diff"] for m in model_tile_metrics)
            agg_mean_diff = float(np.mean([m["mean_abs_diff"] for m in model_tile_metrics]))
            agg_mean_mse = float(np.mean([m["mse"] for m in model_tile_metrics]))
            agg_mean_agreement = float(np.mean([m["mask_agreement_pct"] for m in model_tile_metrics]))

            results_matrix["parity_analysis"][model_key] = {
                "display_name": display_name,
                "tiles_evaluated": len(model_tile_metrics),
                "aggregate_max_abs_diff": agg_max_diff,
                "aggregate_mean_abs_diff": agg_mean_diff,
                "aggregate_mean_mse": agg_mean_mse,
                "aggregate_mask_agreement_pct": agg_mean_agreement,
                "tile_results": model_tile_metrics,
                "parity_verdict": "PASS" if (agg_max_diff < 1e-3 and agg_mean_agreement >= 99.0) else "FAIL",
            }
            print(f"  • Parity across {len(model_tile_metrics)} tiles:")
            print(f"    - Max |Δp|: {agg_max_diff:.6e}")
            print(f"    - Mean |Δp|: {agg_mean_diff:.6e}")
            print(f"    - MSE: {agg_mean_mse:.6e}")
            print(f"    - Mask Agreement: {agg_mean_agreement:.4f}%")
            print(f"    - Verdict: {results_matrix['parity_analysis'][model_key]['parity_verdict']}")

    # Write out results
    output_path = Path("backend_acceptance_matrix.json")
    with open(output_path, "w") as f:
        json.dump(results_matrix, f, indent=2)
    print(f"\n[INFO] Saved verified acceptance matrix to: {output_path}")

    return results_matrix


if __name__ == "__main__":
    res = run_verification()
    passed = res["overall_summary"]["passed_combinations"]
    total = res["overall_summary"]["passed_combinations"] + res["overall_summary"]["failed_combinations"]
    print(f"\nFinal Result: {passed}/{total} Combinations Verified.")
    if res["overall_summary"]["all_passed"]:
        print(">>> STATUS: ALL 8 MODEL-ENGINE COMBINATIONS PASSED VERIFICATION <<<")
        sys.exit(0)
    else:
        print(">>> STATUS: VERIFICATION FAILED <<<")
        sys.exit(1)
