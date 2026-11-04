"""
Unit & Integration Tests for 2026 Advanced Water Segmentation Improvements
Verifies advanced loss formulations, numerical stability, dataset loaders,
and model inference compatibility.
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path

_exp_path = Path(__file__).resolve().parent.parent / "experiments" / "2026_water_segmentation_improvement"
sys.path.insert(0, str(_exp_path))

from src.loss import (
    WeightedBCELoss,
    TverskyLoss,
    FocalTverskyLoss,
    ComboRecallLoss
)
from dataset_advanced import (
    AdvancedSatelliteDataset,
    get_advanced_dataloader
)
from src.models import build_model


def test_weighted_bce_loss_forward():
    loss_fn = WeightedBCELoss(pos_weight=3.0)
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None


def test_tversky_loss_recall_prioritization():
    loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert 0.0 <= loss.item() <= 1.0

    loss.backward()
    assert logits.grad is not None


def test_focal_tversky_loss_empty_mask_stability():
    loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.333)
    # Empty mask test case
    logits = torch.full((1, 1, 32, 32), -10.0) # confident negative
    targets = torch.zeros((1, 1, 32, 32))

    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert 0.0 <= loss.item() <= 1.0


def test_combo_recall_loss_forward():
    loss_fn = ComboRecallLoss(bce_weight=0.3, ft_weight=0.7, pos_weight=2.5)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 32, 32)).float()

    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None


def test_advanced_satellite_dataset_loading():
    ds = AdvancedSatelliteDataset(split="val", augment=False)
    assert len(ds) == 54
    sample = ds[0]
    assert "image" in sample and "mask" in sample and "stem" in sample
    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
    assert sample["image"].min() >= 0.0 and sample["image"].max() <= 1.0


def test_advanced_dataloader_batching():
    loader = get_advanced_dataloader(split="val", batch_size=4, augment=False)
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 256, 256)
    assert batch["mask"].shape == (4, 1, 256, 256)
