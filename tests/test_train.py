"""
GeoVision AI Satellite Water Body Segmentation Platform
Unit Tests for Model Training Module
"""

import pytest
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.train import WaterSegmentationLightningModule, seed_everything


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for training tests.")
def test_seed_everything():
    """Test deterministic seed initialization."""
    seed_everything(42)
    t1 = torch.randn(2, 2)
    seed_everything(42)
    t2 = torch.randn(2, 2)
    assert torch.allclose(t1, t2)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for training tests.")
def test_lightning_module_initialization():
    """Test LightningModule initialization and forward pass."""
    module = WaterSegmentationLightningModule(model_name="unet_plusplus")
    assert module.model is not None

    x = torch.randn(2, 3, 64, 64)
    logits = module(x)
    assert logits.shape == (2, 1, 64, 64)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for training tests.")
def test_training_step_computation():
    """Test training step loss computation."""
    module = WaterSegmentationLightningModule(model_name="unet_plusplus")
    batch = {
        "image": torch.randn(2, 3, 32, 32),
        "mask": torch.randint(0, 2, (2, 1, 32, 32)).float()
    }
    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
