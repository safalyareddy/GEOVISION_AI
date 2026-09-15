"""
GeoVision AI Satellite Water Body Segmentation Platform
C8 Unit Tests: Segmentation Models and Hybrid Loss Layer
"""

from pathlib import Path
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models import build_model, SUPPORTED_MODELS
from src.loss import DiceLoss, FocalLoss, HybridDiceFocalLoss


def test_supported_models_keys():
    """Verify supported model dictionary contains expected architecture keys."""
    assert "unet_plusplus" in SUPPORTED_MODELS
    assert "deeplabv3_plus" in SUPPORTED_MODELS
    assert "segformer" in SUPPORTED_MODELS


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_build_model_unet_plusplus():
    """Test building UNet++ model and executing forward pass."""
    model = build_model("unet_plusplus", in_channels=3, num_classes=1, encoder_weights=None)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_build_model_deeplabv3_plus():
    """Test building DeepLabV3+ model and executing forward pass."""
    model = build_model("deeplabv3_plus", in_channels=3, num_classes=1, encoder_weights=None)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_build_model_segformer():
    """Test building SegFormer model and executing forward pass."""
    model = build_model("segformer", in_channels=3, num_classes=1, encoder_weights=None)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_invalid_model_name_raises_value_error():
    """Test invalid model name raises ValueError."""
    with pytest.raises(ValueError, match="Invalid model name"):
        _ = build_model("unknown_network")


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_dice_loss_forward():
    """Test DiceLoss forward calculation."""
    loss_fn = DiceLoss(from_logits=True)
    preds = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 32, 32))

    loss = loss_fn(preds, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_focal_loss_forward():
    """Test FocalLoss forward calculation."""
    loss_fn = FocalLoss(from_logits=True)
    preds = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 32, 32))

    loss = loss_fn(preds, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_hybrid_loss_forward_and_backward():
    """Test HybridDiceFocalLoss forward calculation and backward gradient computation."""
    loss_fn = HybridDiceFocalLoss(dice_weight=0.5, focal_weight=0.5, from_logits=True)
    preds = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 32, 32))

    loss = loss_fn(preds, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)

    loss.backward()
    assert preds.grad is not None
    assert not torch.isnan(preds.grad).any()
