"""
GeoVision AI Satellite Water Body Segmentation Platform
C11: Inference Verification and PyTorch vs. ONNX Benchmark Script

This script loads PyTorch models and exported ONNX graphs, executes inference,
verifies numeric parity between outputs, and measures performance latency.
"""

import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from src.models import build_model


def compare_pytorch_and_onnx_inference(
    model_name: str = "unet_plusplus",
    onnx_path: str = "checkpoints/weights.onnx",
    checkpoint_path: Optional[str] = None,
    img_size: int = 256,
    num_runs: int = 20,
    atol: float = 1e-3
) -> Dict[str, Any]:
    """
    Run inference using PyTorch and ONNX Runtime, compare outputs for numerical parity,
    and measure execution speeds.
    """
    print("=" * 60)
    print(f"GeoVision AI — PyTorch vs. ONNX Runtime Comparison ({model_name.upper()})")
    print("=" * 60)

    result: Dict[str, Any] = {
        "model_name": model_name,
        "pytorch_latency_ms": 0.0,
        "onnx_latency_ms": 0.0,
        "speedup_factor": 1.0,
        "max_abs_diff": 0.0,
        "outputs_match": True
    }

    dummy_np = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # 1. Run PyTorch Inference
    torch_out_np = None
    if HAS_TORCH:
        try:
            model = build_model(model_name=model_name, encoder_weights=None)
            if checkpoint_path and Path(checkpoint_path).exists():
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict, strict=False)
                print(f"[INFO] Loaded checkpoint weights from '{checkpoint_path}'.")
            model.eval()
            input_tensor = torch.from_numpy(dummy_np)

            # Warmup
            with torch.no_grad():
                _ = model(input_tensor)

            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_runs):
                    torch_out = model(input_tensor)
            torch_latency = ((time.perf_counter() - start) / num_runs) * 1000.0
            result["pytorch_latency_ms"] = float(torch_latency)
            torch_out_np = torch_out.cpu().numpy()
            print(f"[INFO] PyTorch CPU Mean Latency:  {torch_latency:.2f} ms")
        except Exception as e:
            print(f"[WARNING] PyTorch inference execution failed: {e}")

    # 2. Run ONNX Runtime Inference
    onnx_file = Path(onnx_path)
    if HAS_ORT and onnx_file.exists():
        try:
            session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            # Warmup
            _ = session.run([output_name], {input_name: dummy_np})

            start = time.perf_counter()
            for _ in range(num_runs):
                ort_out = session.run([output_name], {input_name: dummy_np})
            ort_latency = ((time.perf_counter() - start) / num_runs) * 1000.0
            result["onnx_latency_ms"] = float(ort_latency)
            ort_out_np = ort_out[0]
            print(f"[INFO] ONNX Runtime Mean Latency: {ort_latency:.2f} ms")

            if result["pytorch_latency_ms"] > 0:
                speedup = result["pytorch_latency_ms"] / ort_latency
                result["speedup_factor"] = float(speedup)
                print(f"[INFO] ONNX Runtime Speedup:      {speedup:.2f}x")

            if torch_out_np is not None:
                max_diff = float(np.max(np.abs(torch_out_np - ort_out_np)))
                result["max_abs_diff"] = max_diff
                matches = bool(max_diff <= atol)
                result["outputs_match"] = matches
                print(f"[INFO] Max Absolute Output Diff:  {max_diff:.6f}")
                print(f"[INFO] Output Parity Check:       {'PASSED' if matches else 'FAILED'}")

        except Exception as e:
            print(f"[WARNING] ONNX Runtime execution failed: {e}")
    else:
        print(f"[INFO] ONNX model file '{onnx_path}' not found or ONNX Runtime unavailable.")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoVision AI Inference Debug Script")
    parser.add_argument("--model", type=str, default="unet_plusplus", help="Model architecture")
    parser.add_argument("--onnx", type=str, default="checkpoints/weights.onnx", help="ONNX model path")
    parser.add_argument("--checkpoint", type=str, default=None, help="PyTorch checkpoint path")
    parser.add_argument("--size", type=int, default=256, help="Input spatial size")
    parser.add_argument("--runs", type=int, default=20, help="Benchmark run count")

    args = parser.parse_args()
    compare_pytorch_and_onnx_inference(
        model_name=args.model,
        onnx_path=args.onnx,
        checkpoint_path=args.checkpoint,
        img_size=args.size,
        num_runs=args.runs
    )
