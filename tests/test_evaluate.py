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
