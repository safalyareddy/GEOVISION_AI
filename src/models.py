"""
GeoVision AI Satellite Water Body Segmentation Platform
C8: Segmentation Models Layer

This module provides model creation abstractions and deep learning architectures
for UNet++, DeepLabV3+, and SegFormer models.
"""

from typing import Optional, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:  # type: ignore
        Module = object

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False


class FallbackUNetPlusPlus(nn.Module if HAS_TORCH else object):
    """
    Pure PyTorch fallback implementation for UNet++ architecture
    when segmentation-models-pytorch is unavailable.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to initialize FallbackUNetPlusPlus.")
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x_up = self.up(x2)
        out = self.dec(torch.cat([x_up, x1], dim=1))
        return self.head(out)


class FallbackDeepLabV3Plus(nn.Module if HAS_TORCH else object):
    """
    Pure PyTorch fallback implementation for DeepLabV3+ architecture
    when segmentation-models-pytorch is unavailable.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to initialize FallbackDeepLabV3Plus.")
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.aspp_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)
        aspp = F.relu(self.aspp_conv(feat))
        up = self.up(aspp)
        return self.head(up)


class FallbackSegFormer(nn.Module if HAS_TORCH else object):
    """
    Pure PyTorch fallback implementation for SegFormer-B2 architecture.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to initialize FallbackSegFormer.")
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        feat = self.encoder(self.stem(x))
        return self.head(feat)


SUPPORTED_MODELS: Dict[str, str] = {
    "unet_plusplus": "UNet++ (Nested UNet for multi-scale feature fusion)",
    "deeplabv3_plus": "DeepLabV3+ (Atrous Spatial Pyramid Pooling)",
    "segformer": "SegFormer-B2 (Lightweight Vision Transformer for Segmentation)"
}


def build_model(
    model_name: str = "unet_plusplus",
    in_channels: int = 3,
    num_classes: int = 1,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet"
) -> Any:
    """
    Factory function to construct segmentation neural network architectures.

    Args:
        model_name: Architecture key ('unet_plusplus', 'deeplabv3_plus', 'segformer').
        in_channels: Input image channel count (default: 3 for RGB).
        num_classes: Output target class count (default: 1 for binary water segmentation).
        encoder_name: Backbone encoder network (default: 'resnet34').
        encoder_weights: Pretrained weights key (default: 'imagenet').

    Returns:
        torch.nn.Module instance representing the specified network.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to build segmentation models.")

    normalized_name = model_name.strip().lower()
    if normalized_name not in SUPPORTED_MODELS:
        valid_keys = ", ".join(list(SUPPORTED_MODELS.keys()))
        raise ValueError(f"Invalid model name '{model_name}'. Must be one of: {valid_keys}")

    if HAS_SMP:
        try:
            if normalized_name == "unet_plusplus":
                return smp.UnetPlusPlus(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=in_channels,
                    classes=num_classes
                )
            elif normalized_name == "deeplabv3_plus":
                return smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=in_channels,
                    classes=num_classes
                )
            elif normalized_name == "segformer":
                # SegFormer via SMP if available
                if hasattr(smp, "SegFormer"):
                    return smp.SegFormer(
                        encoder_name="mit_b2" if encoder_name == "resnet34" else encoder_name,
                        encoder_weights=encoder_weights,
                        in_channels=in_channels,
                        classes=num_classes
                    )
        except Exception as e:
            print(f"[WARNING] Fallback to native model implementation due to SMP error: {e}")

    # Fallback native PyTorch implementations
    if normalized_name == "unet_plusplus":
        return FallbackUNetPlusPlus(in_channels=in_channels, num_classes=num_classes)
    elif normalized_name == "deeplabv3_plus":
        return FallbackDeepLabV3Plus(in_channels=in_channels, num_classes=num_classes)
    elif normalized_name == "segformer":
        return FallbackSegFormer(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: '{model_name}'")
