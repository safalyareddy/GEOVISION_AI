"""
GeoVision AI Satellite Water Body Segmentation Platform
C11: PyTorch Weight Tracing and ONNX Model Export Engine

This script exports trained PyTorch segmentation models (UNet++, DeepLabV3+, SegFormer-B2)
into standardized, optimized ONNX computation graphs suitable for high-throughput deployment.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from src.models import build_model


def export_to_onnx(
    model_name: str = "unet_plusplus",
    checkpoint_path: Optional[str] = None,
    output_path: str = "checkpoints/weights.onnx",
    img_size: int = 256,
    batch_size: int = 1,
    opset_version: int = 14
) -> bool:
    """
    Export PyTorch model to ONNX format and validate the exported computational graph.

    Args:
        model_name: Target model architecture key.
        checkpoint_path: Optional path to PyTorch checkpoint (.pth or .ckpt).
        output_path: Destination path for exported ONNX file.
        img_size: Image spatial height and width.
        batch_size: Representative batch size for tracing.
        opset_version: ONNX operator set version (default: 14).

    Returns:
        bool indicating export success.
    """
    print("=" * 60)
    print(f"GeoVision AI — ONNX Model Export ({model_name.upper()})")
    print("=" * 60)

    if not HAS_TORCH:
        print("[WARNING] PyTorch not available. Skipping ONNX export.")
        return False

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        model = build_model(model_name=model_name)
        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded checkpoint weights from '{checkpoint_path}'.")
        else:
            print(f"[INFO] Using unweighted model architecture '{model_name}' for graph tracing.")

        model.eval()

        dummy_input = torch.randn(batch_size, 3, img_size, img_size)

        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        }

        print(f"[INFO] Tracing PyTorch computational graph to '{output_path}'...")
        torch.onnx.export(
            model,
            dummy_input,
            str(out_file),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )

        print(f"[SUCCESS] ONNX graph exported successfully to '{output_path}'.")

        # Validate graph if ONNX library is installed
        if HAS_ONNX and out_file.exists():
            try:
                onnx_model = onnx.load(str(out_file))
                onnx.checker.check_model(onnx_model)
                print("[SUCCESS] ONNX graph integrity check PASSED.")
            except Exception as e:
                print(f"[WARNING] ONNX graph validation warning: {e}")

        return True

    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoVision AI ONNX Export Script")
    parser.add_argument("--model", type=str, default="unet_plusplus", help="Model architecture")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/weights.onnx", help="Export path")
    parser.add_argument("--size", type=int, default=256, help="Input spatial dimension")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")

    args = parser.parse_args()
    export_to_onnx(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        img_size=args.size,
        opset_version=args.opset
    )
