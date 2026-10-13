"""
GeoVision AI Satellite Water Body Segmentation Platform
C10 Unit Tests: Model Evaluation and Benchmarking Suite
"""

import pytest
import numpy as np

from src.evaluate import calculate_metrics, count_parameters, ModelEvaluator, run_benchmark_suite
from src.models import build_model, SUPPORTED_MODELS


def test_calculate_metrics_perfect_match():
    """Test calculate_metrics with perfect prediction overlap."""
    preds = np.ones((1, 1, 32, 32), dtype=np.float32)
    targets = np.ones((1, 1, 32, 32), dtype=np.uint8)

    metrics = calculate_metrics(preds, targets)
    assert abs(metrics["dice"] - 1.0) < 1e-4
    assert abs(metrics["iou"] - 1.0) < 1e-4
    assert abs(metrics["precision"] - 1.0) < 1e-4
    assert abs(metrics["recall"] - 1.0) < 1e-4


def test_calculate_metrics_no_overlap():
    """Test calculate_metrics when predictions and targets do not overlap."""
    preds = np.zeros((1, 1, 32, 32), dtype=np.float32)
    targets = np.ones((1, 1, 32, 32), dtype=np.uint8)

    metrics = calculate_metrics(preds, targets)
    assert metrics["dice"] < 1e-4
    assert metrics["iou"] < 1e-4


def test_calculate_metrics_hand_computable_partial_overlap():
    """Hand-computable 2x2 grid: TP=1, FP=1, FN=1, TN=1."""
    # Predictions: [1, 1, 0, 0], Targets: [1, 0, 1, 0]
    preds = np.array([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=np.float32)
    targets = np.array([[[[1, 0], [1, 0]]]], dtype=np.uint8)

    metrics = calculate_metrics(preds, targets, threshold=0.5)
    # Dice = 2*1 / (2 + 2) = 0.50
    assert abs(metrics["dice"] - 0.5) < 1e-4
    # IoU = 1 / (2 + 2 - 1) = 1/3
    assert abs(metrics["iou"] - (1.0 / 3.0)) < 1e-4
    # Precision = 1 / 2 = 0.5
    assert abs(metrics["precision"] - 0.5) < 1e-4
    # Recall = 1 / 2 = 0.5
    assert abs(metrics["recall"] - 0.5) < 1e-4


def test_calculate_metrics_empty_mask_handling():
    """Verify Laplace smoothing conventions on empty masks."""
    # Case 1: Both empty (true negative tile) -> Dice = 1.0
    preds_empty = np.zeros((1, 1, 4, 4), dtype=np.float32)
    targets_empty = np.zeros((1, 1, 4, 4), dtype=np.uint8)
    metrics_tn = calculate_metrics(preds_empty, targets_empty)
    assert abs(metrics_tn["dice"] - 1.0) < 1e-4
    assert abs(metrics_tn["iou"] - 1.0) < 1e-4

    # Case 2: Target empty, prediction non-empty (false positive tile) -> Dice ~ 0.0
    preds_fp = np.ones((1, 1, 4, 4), dtype=np.float32)
    metrics_fp = calculate_metrics(preds_fp, targets_empty)
    assert metrics_fp["dice"] < 1e-4
    assert metrics_fp["iou"] < 1e-4


def test_model_evaluator_execution():
    """Test ModelEvaluator for single model architecture."""
    evaluator = ModelEvaluator(model_name="unet_plusplus")
    res = evaluator.evaluate_model()

    assert "model_name" in res
    assert res["model_name"] == "unet_plusplus"
    assert "dice" in res
    assert "iou" in res
    assert "latency_ms" in res
    assert "fps" in res


def test_run_benchmark_suite_returns_all_models():
    """Test benchmark suite runs across all supported architectures."""
    results = run_benchmark_suite(save_csv=False)
    assert len(results) == len(SUPPORTED_MODELS)
    model_names = [r["model_name"] for r in results]
    for key in SUPPORTED_MODELS:
        assert key in model_names
