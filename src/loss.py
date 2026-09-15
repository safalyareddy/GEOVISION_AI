"""
GeoVision AI Satellite Water Body Segmentation Platform
C8: Loss Functions Layer

This module implements custom loss functions tailored for binary satellite
image segmentation under extreme class imbalance (land vs water).
"""

from typing import Optional, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    Tensor = torch.Tensor
except ImportError:
    HAS_TORCH = False
    Tensor = Any  # type: ignore
    class nn:  # type: ignore
        Module = object


class DiceLoss(nn.Module if HAS_TORCH else object):
    """
    Dice Loss implementation for binary spatial segmentation.
    Calculates overlap between predicted probabilities and ground truth masks.
    """
    def __init__(self, smooth: float = 1e-6, from_logits: bool = True):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to initialize DiceLoss.")
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self.from_logits:
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs

        probs = probs.view(-1)
        targets = targets.view(-1).float()

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.pow(2).sum() + targets.pow(2).sum() + self.smooth)
        return 1.0 - dice


class FocalLoss(nn.Module if HAS_TORCH else object):
    """
    Focal Loss implementation for handling severe class imbalance.
    Down-weights well-classified background (land) pixels to focus on hard foreground (water).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, from_logits: bool = True):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to initialize FocalLoss.")
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
            probs = torch.sigmoid(inputs)
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets.float(), reduction="none")
            probs = inputs

        p_t = probs * targets.float() + (1.0 - probs) * (1.0 - targets.float())
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)
        alpha_factor = self.alpha * targets.float() + (1.0 - self.alpha) * (1.0 - targets.float())
        focal_weight = alpha_factor * torch.pow(1.0 - p_t, self.gamma)

        focal_loss = focal_weight * bce_loss
        return focal_loss.mean()


class HybridDiceFocalLoss(nn.Module if HAS_TORCH else object):
    """
    Hybrid Joint Loss combining Dice Loss and Focal Loss:
    L_total = alpha * L_Dice + beta * L_Focal
    """
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        from_logits: bool = True
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to initialize HybridDiceFocalLoss.")
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(from_logits=from_logits)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, from_logits=from_logits)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        l_dice = self.dice_loss(inputs, targets)
        l_focal = self.focal_loss(inputs, targets)
        return self.dice_weight * l_dice + self.focal_weight * l_focal
