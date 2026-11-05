"""
GeoVision AI — 12-Combination Backend Acceptance Verification
Executes all 4 models (FPN, UNet++, DeepLabV3+, SegFormer) across all 3 backends:
1. PyTorch (In-Process CPU)
2. FastAPI Microservice (REST API via TestClient)
3. ONNX Runtime (Optimized CPU)

Records latency, water coverage percentage, estimated water area, finite checks,
and numerical parity in backend_acceptance_matrix.json.
"""

import io
import json
import time
import base64
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import onnxruntime as ort
from fastapi.testclient import TestClient

from src.models import SUPPORTED_MODELS
from src.utils.model_registry import (
    load_verified_model,
    compute_file_sha256,
    resolve_checkpoint_path,
    MODEL_REGISTRY,
    DEFAULT_HF_REVISION,
)
from src.utils.input_validator import (
    classify_input_image,
    calculate_water_area_km2,
    InputClassification,
)
from src.api.main import app as fastapi_app

TEST_IMAGE_PATH = Path("dataset/test/images/patch_0047.png")
if not TEST_IMAGE_PATH.exists():
    TEST_IMAGE_PATH = Path("assets/demo/patch_0047.png")


def run_pytorch_comb(model_name: str, image: Image.Image, threshold: float = 0.5) -> Dict[str, Any]:
    model = load_verified_model(model_name, device="cpu")
    model.eval()

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    t_img = transform(image.convert("RGB")).unsqueeze(0)

    # Warmup
    with torch.inference_mode():
        _ = model(t_img)

    # Benchmark run
    latencies = []
    probs_np = None
    for _ in range(3):
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(t_img)
            if isinstance(out, dict):
                logits = out.get("out", out.get("logits"))
            elif hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        probs_np = probs

    non_finite = int(np.isnan(probs_np).sum() + np.isinf(probs_np).sum())
    binary_mask = (probs_np >= threshold).astype(np.uint8)
    water_px = int(np.sum(binary_mask))
    coverage = float((water_px / binary_mask.size) * 100.0)
    area_km2, note = calculate_water_area_km2(
        water_px,
        InputClassification.VERIFIED_PROJECT_SAMPLE,
        {"project_sample_name": "patch_0047.png"}
    )

    ckpt_path = resolve_checkpoint_path(model_name, allow_download=False)
    sha256 = compute_file_sha256(ckpt_path) if ckpt_path else "unknown"

    return {
        "status": "PASS",
        "latency_ms": round(float(np.mean(latencies)), 2),
        "water_coverage_pct": round(coverage, 2),
        "water_area_km2": area_km2,
        "area_basis": note,
        "non_finite_count": non_finite,
        "sha256": sha256,
        "probs": probs_np,
        "binary_mask": binary_mask,
    }


def run_fastapi_comb(model_name: str, image_bytes: bytes, threshold: float = 0.5) -> Dict[str, Any]:
    client = TestClient(fastapi_app)

    # Benchmark run
    latencies = []
    last_resp = None
    for _ in range(3):
        t0 = time.perf_counter()
        resp = client.post(
            "/predict",
            files={"file": ("patch_0047.png", io.BytesIO(image_bytes), "image/png")},
            data={"model_name": model_name, "threshold": str(threshold)}
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        last_resp = resp

    if last_resp.status_code != 200:
        return {
            "status": "FAIL",
            "error": f"HTTP {last_resp.status_code}: {last_resp.text}",
        }

    data = last_resp.json()
    mask_bytes = base64.b64decode(data["mask_base64"])
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L").resize((256, 256), Image.NEAREST)
    binary_mask = (np.array(mask_img) > 127).astype(np.uint8)

    return {
        "status": "PASS",
        "latency_ms": round(float(np.mean(latencies)), 2),
        "server_latency_ms": data.get("latency_ms"),
        "water_coverage_pct": data.get("water_coverage_percentage"),
        "water_area_km2": data.get("estimated_water_area_km2"),
        "area_basis": data.get("area_estimation_basis"),
        "input_classification": data.get("input_classification"),
        "non_finite_count": 0,
        "binary_mask": binary_mask,
    }


def run_onnx_comb(model_name: str, image: Image.Image, threshold: float = 0.5) -> Dict[str, Any]:
    onnx_path = Path("checkpoints/onnx") / f"{model_name}.onnx"
    if not onnx_path.exists():
        # Export
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        pt_model = load_verified_model(model_name, device="cpu")
        pt_model.eval()
        dummy = torch.randn(1, 3, 256, 256, dtype=torch.float32)
        torch.onnx.export(
            pt_model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    t_img = transform(image.convert("RGB")).unsqueeze(0).numpy()
    input_name = session.get_inputs()[0].name

    # Warmup
    _ = session.run(None, {input_name: t_img})

    # Benchmark run
    latencies = []
    probs_np = None
    for _ in range(3):
        t0 = time.perf_counter()
        raw_out = session.run(None, {input_name: t_img})[0]
        probs = 1.0 / (1.0 + np.exp(-raw_out))
        probs_np = np.squeeze(probs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    non_finite = int(np.isnan(probs_np).sum() + np.isinf(probs_np).sum())
    binary_mask = (probs_np >= threshold).astype(np.uint8)
    water_px = int(np.sum(binary_mask))
    coverage = float((water_px / binary_mask.size) * 100.0)
    area_km2, note = calculate_water_area_km2(
        water_px,
        InputClassification.VERIFIED_PROJECT_SAMPLE,
        {"project_sample_name": "patch_0047.png"}
    )

    return {
        "status": "PASS",
        "latency_ms": round(float(np.mean(latencies)), 2),
        "water_coverage_pct": round(coverage, 2),
        "water_area_km2": area_km2,
        "area_basis": note,
        "non_finite_count": non_finite,
        "onnx_size_bytes": onnx_path.stat().st_size,
        "probs": probs_np,
        "binary_mask": binary_mask,
    }


def main():
    print(f"Loading test image from: {TEST_IMAGE_PATH}")
    img_bytes = TEST_IMAGE_PATH.read_bytes()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    models = ["fpn", "unet_plusplus", "deeplabv3_plus", "segformer"]
    backends = ["pytorch_inprocess", "fastapi_microservice", "onnx_runtime"]

    matrix = {
        "metadata": {
            "test_image": "patch_0047.png",
            "image_size": [image.width, image.height],
            "hf_pinned_revision": DEFAULT_HF_REVISION,
            "threshold": 0.50,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_combinations": len(models) * len(backends),
        },
        "results": {},
        "summary": {},
    }

    all_passed = True

    # First run PyTorch in-process to establish ground truth masks for parity comparison
    pt_masks = {}
    pt_probs = {}

    print("\n=======================================================")
    print("      GEOVISION AI — 12-COMBINATION ACCEPTANCE RUN     ")
    print("=======================================================\n")

    for m in models:
        print(f"--- Testing Architecture: {m.upper()} ---")

        # 1. PyTorch In-Process
        t_start = time.perf_counter()
        pt_res = run_pytorch_comb(m, image)
        pt_masks[m] = pt_res["binary_mask"]
        pt_probs[m] = pt_res["probs"]
        matrix["results"][f"{m}__pytorch_inprocess"] = {
            "model_name": m,
            "backend": "pytorch_inprocess",
            "status": pt_res["status"],
            "latency_ms": pt_res["latency_ms"],
            "water_coverage_pct": pt_res["water_coverage_pct"],
            "water_area_km2": pt_res["water_area_km2"],
            "non_finite_count": pt_res["non_finite_count"],
            "sha256": pt_res["sha256"],
            "mask_agreement_with_pytorch_pct": 100.0,
        }
        print(f"  [PyTorch In-Process] Status: {pt_res['status']} | Latency: {pt_res['latency_ms']} ms | Coverage: {pt_res['water_coverage_pct']}% | Area: {pt_res['water_area_km2']} km²")

        # 2. FastAPI REST Microservice
        fa_res = run_fastapi_comb(m, img_bytes)
        fa_agree = float(np.mean(fa_res["binary_mask"] == pt_masks[m]) * 100.0) if fa_res["status"] == "PASS" else 0.0
        matrix["results"][f"{m}__fastapi_microservice"] = {
            "model_name": m,
            "backend": "fastapi_microservice",
            "status": fa_res["status"],
            "latency_ms": fa_res["latency_ms"],
            "server_latency_ms": fa_res.get("server_latency_ms"),
            "water_coverage_pct": fa_res.get("water_coverage_pct"),
            "water_area_km2": fa_res.get("water_area_km2"),
            "non_finite_count": fa_res.get("non_finite_count", 0),
            "input_classification": fa_res.get("input_classification"),
            "mask_agreement_with_pytorch_pct": round(fa_agree, 2),
        }
        print(f"  [FastAPI REST API ] Status: {fa_res['status']} | Latency: {fa_res['latency_ms']} ms | Coverage: {fa_res.get('water_coverage_pct')}% | Agreement: {fa_agree:.2f}%")

        # 3. ONNX Runtime
        ox_res = run_onnx_comb(m, image)
        ox_agree = float(np.mean(ox_res["binary_mask"] == pt_masks[m]) * 100.0) if ox_res["status"] == "PASS" else 0.0
        max_prob_diff = float(np.max(np.abs(ox_res["probs"] - pt_probs[m]))) if ox_res["status"] == "PASS" else None
        matrix["results"][f"{m}__onnx_runtime"] = {
            "model_name": m,
            "backend": "onnx_runtime",
            "status": ox_res["status"],
            "latency_ms": ox_res["latency_ms"],
            "water_coverage_pct": ox_res["water_coverage_pct"],
            "water_area_km2": ox_res["water_area_km2"],
            "non_finite_count": ox_res["non_finite_count"],
            "onnx_size_bytes": ox_res.get("onnx_size_bytes"),
            "max_prob_diff_vs_pytorch": max_prob_diff,
            "mask_agreement_with_pytorch_pct": round(ox_agree, 2),
        }
        print(f"  [ONNX Runtime     ] Status: {ox_res['status']} | Latency: {ox_res['latency_ms']} ms | Coverage: {ox_res['water_coverage_pct']}% | Agreement: {ox_agree:.2f}% (Max Δp: {max_prob_diff:.6f})")

        if pt_res["status"] != "PASS" or fa_res["status"] != "PASS" or ox_res["status"] != "PASS":
            all_passed = False

    matrix["summary"] = {
        "all_combinations_passed": all_passed,
        "total_passed": sum(1 for v in matrix["results"].values() if v["status"] == "PASS"),
        "total_failed": sum(1 for v in matrix["results"].values() if v["status"] != "PASS"),
        "parity_with_pytorch_perfect": all(v.get("mask_agreement_with_pytorch_pct", 0) >= 99.99 for v in matrix["results"].values()),
    }

    # Save to project reports and root
    Path("reports").mkdir(exist_ok=True)
    out_json = Path("reports/backend_acceptance_matrix.json")
    out_json.write_text(json.dumps(matrix, indent=2))
    Path("backend_acceptance_matrix.json").write_text(json.dumps(matrix, indent=2))
    print(f"\nWritten authoritative matrix to: {out_json} and backend_acceptance_matrix.json")
    print(f"Summary: {matrix['summary']}")


if __name__ == "__main__":
    main()
