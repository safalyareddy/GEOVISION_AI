#!/usr/bin/env python3
"""
GeoVision AI — Reproducible Four-Model Scientific Benchmark Comparison
Evaluates FPN, UNet++, DeepLabV3+, and SegFormer across the 54 held-out test split.
Calculates Micro/Macro Dice, IoU, Precision, Recall, Confusion Matrix, Latency, and Coverage.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from src.models import SUPPORTED_MODELS
from src.inference import load_model, STANDARD_INPUT_SIZE, NOMINAL_GSD_METERS
from src.utils.model_registry import (
    MODEL_REGISTRY,
    compute_file_sha256,
    resolve_checkpoint_path
)

PARAM_COUNTS = {
    "fpn": 23155393,
    "unet_plusplus": 26078609,
    "deeplabv3_plus": 22437457,
    "segformer": 3714401,
}

# Thresholds established on the independent VALIDATION set during experiment runs
VALIDATION_OPTIMAL_THRESHOLDS = {
    "fpn": 0.60,
    "unet_plusplus": 0.60,
    "deeplabv3_plus": 0.60,
    "segformer": 0.55,
}


def calculate_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    """Calculate pixel-level confusion matrix and derived single-image metrics."""
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))

    # Per-image Dice & IoU
    denom_dice = 2.0 * tp + fp + fn
    dice = (2.0 * tp / denom_dice) if denom_dice > 0 else (1.0 if (tp + fn == 0 and fp == 0) else 0.0)

    denom_iou = tp + fp + fn
    iou = (tp / denom_iou) if denom_iou > 0 else (1.0 if (tp + fn == 0 and fp == 0) else 0.0)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "dice": float(dice), "iou": float(iou)
    }


def evaluate_model_on_split(
    model_name: str,
    threshold: float,
    dataset_dir: Path,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Run rigorous benchmark evaluation of a single model on a dataset split."""
    img_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"

    img_files = sorted(list(img_dir.glob("*.png")))
    if not img_files:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")

    transform = T.Compose([
        T.Resize(STANDARD_INPUT_SIZE),
        T.ToTensor()
    ])

    model = load_model(model_name, device=device)
    model.eval()

    tot_tp, tot_fp, tot_fn, tot_tn = 0, 0, 0, 0
    total_pixels = 0
    total_pred_water = 0
    total_gt_water = 0

    per_image_dice: List[float] = []
    per_image_iou: List[float] = []
    per_image_coverage: List[float] = []
    latencies: List[float] = []

    with torch.inference_mode():
        for img_p in img_files:
            msk_p = mask_dir / img_p.name
            if not msk_p.exists():
                raise FileNotFoundError(f"Missing ground-truth mask for {img_p.name}")

            img = Image.open(img_p).convert("RGB")
            gt = Image.open(msk_p).convert("L")
            gt_arr = (np.array(gt.resize(STANDARD_INPUT_SIZE, resample=Image.NEAREST)) > 127).astype(np.uint8)

            x = transform(img).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            out = model(x)
            if isinstance(out, dict):
                logits = out.get("out", out.get("logits"))
            elif hasattr(out, "logits"):
                logits = out.logits
            else:
                logits = out

            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            latencies.append((time.perf_counter() - t0) * 1000.0)

            pred_bin = (probs >= threshold).astype(np.uint8)

            m = calculate_binary_metrics(pred_bin, gt_arr)
            tot_tp += m["tp"]
            tot_fp += m["fp"]
            tot_fn += m["fn"]
            tot_tn += m["tn"]

            per_image_dice.append(m["dice"])
            per_image_iou.append(m["iou"])

            pred_water_px = int(np.sum(pred_bin))
            gt_water_px = int(np.sum(gt_arr))
            total_pred_water += pred_water_px
            total_gt_water += gt_water_px
            total_pixels += pred_bin.size

            per_image_coverage.append((pred_water_px / pred_bin.size) * 100.0)

    # Micro Aggregate Metrics
    pixel_acc = (tot_tp + tot_tn) / total_pixels
    precision = (tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) > 0 else 0.0
    recall = (tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) > 0 else 0.0
    micro_dice = (2.0 * tot_tp / (2 * tot_tp + tot_fp + tot_fn)) if (2 * tot_tp + tot_fp + tot_fn) > 0 else 0.0
    micro_iou = (tot_tp / (tot_tp + tot_fp + tot_fn)) if (tot_tp + tot_fp + tot_fn) > 0 else 0.0

    # Macro Averages
    macro_dice = float(np.mean(per_image_dice))
    macro_iou = float(np.mean(per_image_iou))
    f1_score = micro_dice

    mean_pred_cov = float(np.mean(per_image_coverage))
    total_pred_cov = float((total_pred_water / total_pixels) * 100.0)
    total_gt_cov = float((total_gt_water / total_pixels) * 100.0)
    avg_latency_ms = float(np.mean(latencies))

    ckpt_path = resolve_checkpoint_path(model_name, allow_download=False)
    ckpt_sha = compute_file_sha256(ckpt_path) if ckpt_path else None

    return {
        "model_name": model_name,
        "display_name": MODEL_REGISTRY[model_name]["display_name"],
        "threshold": float(threshold),
        "is_champion": MODEL_REGISTRY[model_name]["is_champion"],
        "parameter_count": PARAM_COUNTS.get(model_name, 0),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_path": str(ckpt_path) if ckpt_path else None,
        "pixel_accuracy": float(pixel_acc),
        "precision": float(precision),
        "recall": float(recall),
        "micro_dice": float(micro_dice),
        "macro_dice": float(macro_dice),
        "micro_iou": float(micro_iou),
        "macro_iou": float(macro_iou),
        "f1_score": float(f1_score),
        "confusion_matrix": {
            "tp": tot_tp,
            "fp": tot_fp,
            "fn": tot_fn,
            "tn": tot_tn
        },
        "per_image_stats": {
            "dice_std": float(np.std(per_image_dice)),
            "dice_median": float(np.median(per_image_dice)),
            "dice_min": float(np.min(per_image_dice)),
            "dice_max": float(np.max(per_image_dice)),
            "iou_std": float(np.std(per_image_iou)),
            "iou_median": float(np.median(per_image_iou))
        },
        "mean_predicted_coverage_pct": round(mean_pred_cov, 2),
        "total_predicted_coverage_pct": round(total_pred_cov, 2),
        "ground_truth_coverage_pct": round(total_gt_cov, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "num_test_tiles": len(img_files)
    }


def run_benchmark(
    dataset_dir: str = "dataset/test",
    output_json: str = "reports/four_model_scientific_benchmark.json",
    device: str = "cpu"
) -> Dict[str, Any]:
    """Execute complete four-model benchmark."""
    d_path = Path(dataset_dir)
    print("=" * 95)
    print("GEOVISION AI — COMPLETE FOUR-MODEL REPRODUCIBLE PYTORCH BENCHMARK")
    print(f"Target Split: {d_path.resolve()} | Device: {device}")
    print("=" * 95)

    models = ["fpn", "unet_plusplus", "deeplabv3_plus", "segformer"]

    results_fixed_th05 = {}
    results_val_calibrated = {}

    print("\n--- SECTION 1: COMMON BENCHMARK (FIXED THRESHOLD tau = 0.50) ---")
    for m in models:
        res = evaluate_model_on_split(m, threshold=0.50, dataset_dir=d_path, device=device)
        results_fixed_th05[m] = res
        print(f"[{m:<14}] Acc: {res['pixel_accuracy']*100:5.2f}% | Prec: {res['precision']*100:5.2f}% | "
              f"Rec: {res['recall']*100:5.2f}% | Micro Dice: {res['micro_dice']:.4f} | "
              f"Macro Dice: {res['macro_dice']:.4f} | Micro IoU: {res['micro_iou']:.4f} | "
              f"Latency: {res['avg_latency_ms']:5.1f}ms")

    print("\n--- SECTION 2: VALIDATION-CALIBRATED BENCHMARK (NON-TEST OPTIMIZED) ---")
    for m in models:
        val_th = VALIDATION_OPTIMAL_THRESHOLDS.get(m, 0.50)
        res_cal = evaluate_model_on_split(m, threshold=val_th, dataset_dir=d_path, device=device)
        results_val_calibrated[m] = res_cal
        print(f"[{m:<14} (tau={val_th:.2f})] Acc: {res_cal['pixel_accuracy']*100:5.2f}% | Prec: {res_cal['precision']*100:5.2f}% | "
              f"Rec: {res_cal['recall']*100:5.2f}% | Micro Dice: {res_cal['micro_dice']:.4f} | "
              f"Macro Dice: {res_cal['macro_dice']:.4f} | Micro IoU: {res_cal['micro_iou']:.4f}")

    benchmark_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_split": str(d_path),
        "num_test_tiles": 54,
        "device": device,
        "fixed_threshold_0_50": results_fixed_th05,
        "validation_calibrated_thresholds": results_val_calibrated
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"\nSaved comprehensive benchmark report to: {out_p}")

    return benchmark_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoVision AI Four-Model Benchmark")
    parser.add_argument("--dataset_dir", type=str, default="dataset/test", help="Path to test dataset directory")
    parser.add_argument("--output", type=str, default="reports/four_model_scientific_benchmark.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device ('cpu', 'mps')")
    args = parser.parse_args()

    run_benchmark(dataset_dir=args.dataset_dir, output_json=args.output, device=args.device)
