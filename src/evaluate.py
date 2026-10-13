"""
GeoVision AI Satellite Water Body Segmentation Platform
C10: Model Evaluation and Benchmarking Suite

This module provides cross-architecture benchmarking utilities to evaluate
segmentation models (UNet++, DeepLabV3+, SegFormer-B2) on Dice, IoU,
Precision, Recall, inference latency, FPS, parameter count, and memory footprint.
"""

import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models import build_model, SUPPORTED_MODELS
from src.data.dataset import SatelliteWaterDataset, get_dataloader


def calculate_metrics(
    pred_probs: np.ndarray,
    target_masks: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute binary segmentation evaluation metrics.

    Args:
        pred_probs: Model output probability array [0, 1].
        target_masks: Ground truth binary mask array {0, 1}.
        threshold: Binary decision threshold (default: 0.5).

    Returns:
        Dictionary containing Dice, IoU, Precision, and Recall scores.
    """
    binary_preds = (pred_probs >= threshold).astype(np.uint8)
    targets = (target_masks >= threshold).astype(np.uint8)

    intersection = np.logical_and(binary_preds, targets).sum()
    total_pred = binary_preds.sum()
    total_target = targets.sum()

    dice = (2.0 * intersection + 1e-6) / (total_pred + total_target + 1e-6)
    iou = (intersection + 1e-6) / (total_pred + total_target - intersection + 1e-6)
    precision = (intersection + 1e-6) / (total_pred + 1e-6)
    recall = (intersection + 1e-6) / (total_target + 1e-6)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall)
    }


def count_parameters(model: Any) -> Tuple[int, float]:
    """
    Calculate model parameter count and estimated memory size in MB.
    """
    if not HAS_TORCH or not isinstance(model, torch.nn.Module):
        return 0, 0.0

    total_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size_bytes + buffer_size_bytes) / (1024 ** 2)

    return total_params, float(size_mb)


class ModelEvaluator:
    """
    Evaluator orchestrating multi-architecture model benchmarks.
    """
    def __init__(self, model_name: str = "unet_plusplus", dataset_dir: str = "dataset"):
        self.model_name = model_name
        self.dataset_dir = dataset_dir

    def evaluate_model(
        self,
        model: Optional[Any] = None,
        dataloader: Optional[Any] = None,
        num_warmup: int = 5,
        num_eval_runs: int = 20
    ) -> Dict[str, Any]:
        """
        Evaluate a single segmentation model architecture.

        Returns benchmark metrics including accuracy scores and latency statistics.
        """
        if model is None and HAS_TORCH:
            model = build_model(model_name=self.model_name)

        total_params, size_mb = count_parameters(model)

        # Benchmark inference latency
        latencies: List[float] = []
        if HAS_TORCH and model is not None:
            model.eval()
            dummy_input = torch.randn(1, 3, 256, 256)

            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model(dummy_input)

                for _ in range(num_eval_runs):
                    start_time = time.perf_counter()
                    _ = model(dummy_input)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    latencies.append(elapsed_ms)

        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        fps = float(1000.0 / mean_latency) if mean_latency > 0 else 0.0

        # Calculate dataset scores
        dice_scores: List[float] = []
        iou_scores: List[float] = []
        precision_scores: List[float] = []
        recall_scores: List[float] = []

        if dataloader is not None and HAS_TORCH and model is not None:
            model.eval()
            for batch in dataloader:
                if isinstance(batch, dict) and "image" in batch and "mask" in batch:
                    img = batch["image"]
                    mask = batch["mask"]
                    if isinstance(img, torch.Tensor):
                        with torch.no_grad():
                            preds = torch.sigmoid(model(img)).cpu().numpy()
                            targets = mask.cpu().numpy()
                        for p, t in zip(preds, targets):
                            metrics = calculate_metrics(p, t)
                            dice_scores.append(metrics["dice"])
                            iou_scores.append(metrics["iou"])
                            precision_scores.append(metrics["precision"])
                            recall_scores.append(metrics["recall"])

        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
        mean_prec = float(np.mean(precision_scores)) if precision_scores else 0.0
        mean_rec = float(np.mean(recall_scores)) if recall_scores else 0.0

        return {
            "model_name": self.model_name,
            "dice": mean_dice,
            "iou": mean_iou,
            "precision": mean_prec,
            "recall": mean_rec,
            "latency_ms": mean_latency,
            "fps": fps,
            "parameters": total_params,
            "size_mb": size_mb
        }


def run_benchmark_suite(
    dataset_dir: str = "dataset",
    checkpoint_dir: str = "checkpoints",
    save_csv: bool = False,
    output_path: str = "benchmark_results.csv"
) -> List[Dict[str, Any]]:
    """
    Run evaluation suite across all supported model architectures.
    """
    print("=" * 60)
    print("GeoVision AI — Cross-Architecture Evaluation Suite")
    print("=" * 60)

    test_loader = None
    if HAS_TORCH:
        try:
            test_loader = get_dataloader(split="test", root_dir=dataset_dir, batch_size=4, shuffle=False)
            print(f"[INFO] Loaded test dataset with {len(test_loader.dataset)} samples from '{dataset_dir}'.")
        except Exception as e:
            print(f"[WARNING] Could not load test dataset from '{dataset_dir}': {e}.")

    results: List[Dict[str, Any]] = []
    ckpt_map = {
        "unet_plusplus": "unet_plusplus_best.pth",
        "deeplabv3_plus": "deeplabv3_plus_best.pth",
        "segformer": "segformer_best.pth",
        "segformer_b2": "segformer_best.pth"
    }

    for model_key in SUPPORTED_MODELS:
        evaluator = ModelEvaluator(model_name=model_key, dataset_dir=dataset_dir)
        model = None
        if HAS_TORCH:
            model = build_model(model_name=model_key)
            ckpt_name = ckpt_map.get(model_key, f"{model_key}_best.pth")
            ckpt_path = Path(checkpoint_dir) / ckpt_name
            if ckpt_path.exists():
                try:
                    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    try:
                        model.load_state_dict(state, strict=True)
                        print(f"[INFO] Loaded checkpoint '{ckpt_path}' for model '{model_key}' (strict=True, 0 missing/unexpected).")
                    except Exception as strict_err:
                        model.load_state_dict(state, strict=False)
                        print(f"[WARNING] Non-strict load fallback for '{ckpt_path}': {strict_err}")
                except Exception as ex:
                    print(f"[WARNING] Failed loading checkpoint '{ckpt_path}': {ex}.")

        res = evaluator.evaluate_model(model=model, dataloader=test_loader)
        results.append(res)

        print(
            f"[INFO] Model: {model_key:<14} | Dice: {res['dice']:.4f} | IoU: {res['iou']:.4f} | "
            f"Latency: {res['latency_ms']:.2f} ms | FPS: {res['fps']:.1f} | Params: {res['parameters'] / 1e6:.2f}M"
        )

    if save_csv and results:
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"[SUCCESS] Benchmark report saved to '{output_path}'.")
        except ImportError:
            print("[WARNING] Pandas required to save CSV benchmark results.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoVision AI Benchmark Runner")
    parser.add_argument("--data-root", type=str, default="dataset", help="Dataset root path")
    parser.add_argument("--save-csv", action="store_true", help="Save benchmark results to CSV")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Output CSV filepath")

    args = parser.parse_args()
    run_benchmark_suite(dataset_dir=args.data_root, save_csv=args.save_csv, output_path=args.output)
