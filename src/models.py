"""
GeoVision AI Satellite Water Body Segmentation Platform
Model Architectures Layer

This module provides production-grade model construction abstractions for
UNet++, DeepLabV3+, and SegFormer architectures.
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

try:
    from transformers import SegformerForSemanticSegmentation, SegformerConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class HuggingFaceSegFormerWrapper(nn.Module if HAS_TORCH else object):
    """
    Production wrapper for Hugging Face SegFormer Vision Transformer.
    Interpolates coarse 4x downsampled decoder logits to full input spatial resolution.
    """
    def __init__(
        self,
        encoder_name: str = "nvidia/mit-b0",
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_weights: Optional[str] = "imagenet"
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to initialize HuggingFaceSegFormerWrapper.")
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "The 'transformers' library is required for SegFormer. "
                "Install via: pip install transformers"
            )
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.encoder_name = encoder_name

        # Determine architecture parameters based on encoder variant (MiT-B2 vs MiT-B0)
        is_b0 = "b0" in encoder_name.lower()
        depths = [2, 2, 2, 2] if is_b0 else [3, 4, 6, 3]
        hidden_sizes = [32, 64, 160, 256] if is_b0 else [64, 128, 320, 512]
        decoder_hidden_size = 256 if is_b0 else 768

        label_mapping = {0: "water"}
        reverse_mapping = {"water": 0}
        if encoder_weights is None:
            self.config = SegformerConfig(
                num_labels=num_classes,
                num_channels=in_channels,
                depths=depths,
                hidden_sizes=hidden_sizes,
                decoder_hidden_size=decoder_hidden_size,
                id2label=label_mapping,
                label2id=reverse_mapping
            )
        else:
            try:
                self.config = SegformerConfig.from_pretrained(
                    encoder_name,
                    num_labels=num_classes,
                    id2label=label_mapping,
                    label2id=reverse_mapping
                )
            except Exception:
                # Fallback to local structural config
                self.config = SegformerConfig(
                    num_labels=num_classes,
                    num_channels=in_channels,
                    depths=depths,
                    hidden_sizes=hidden_sizes,
                    decoder_hidden_size=decoder_hidden_size,
                    id2label=label_mapping,
                    label2id=reverse_mapping
                )
        self.model = SegformerForSemanticSegmentation(self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        out = self.model(x)
        logits = out.logits
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)


class FallbackUNetPlusPlus(nn.Module if HAS_TORCH else object):
    """
    Explicit synthetic UNet++ mock model for testing environments
    where third-party libraries are excluded. Never used silently in production.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x_up = self.up(x2)
        out = self.dec(torch.cat([x_up, x1], dim=1))
        return self.head(out)


SUPPORTED_MODELS: Dict[str, str] = {
    "unet_plusplus": "UNet++–ResNet34 (Nested UNet for multi-scale feature fusion)",
    "deeplabv3_plus": "DeepLabV3+–ResNet34 (Atrous Spatial Pyramid Pooling)",
    "segformer": "SegFormer–MiT-B0 (Lightweight Vision Transformer)",
    "fpn": "FPN–ResNet34 (Champion Feature Pyramid Network)"
}


def build_model(
    model_name: str = "unet_plusplus",
    in_channels: int = 3,
    num_classes: int = 1,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    allow_dummy_fallback: bool = False
) -> Any:
    """
    Factory function to construct segmentation neural network architectures.

    Args:
        model_name: Architecture key ('unet_plusplus', 'deeplabv3_plus', 'segformer', 'fpn').
        in_channels: Input image channel count (default: 3 for RGB).
        num_classes: Output target class count (default: 1 for binary water segmentation).
        encoder_name: Backbone encoder network (default: 'resnet34').
        encoder_weights: Pretrained weights key (default: 'imagenet').
        allow_dummy_fallback: If True, allows synthetic mock fallback for unit tests.
                              If False (default), raises an explicit error on missing dependencies.

    Returns:
        torch.nn.Module instance representing the specified network.

    Raises:
        ValueError: If model_name is unrecognized.
        ImportError: If required framework (SMP or Transformers) is missing.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to build segmentation models.")

    normalized_name = model_name.strip().lower()
    if normalized_name not in SUPPORTED_MODELS:
        valid_keys = ", ".join(list(SUPPORTED_MODELS.keys()))
        raise ValueError(f"Invalid model name '{model_name}'. Must be one of: {valid_keys}")

    if normalized_name in ["unet_plusplus", "deeplabv3_plus", "fpn"]:
        if not HAS_SMP:
            if allow_dummy_fallback:
                return FallbackUNetPlusPlus(in_channels=in_channels, num_classes=num_classes)
            raise ImportError(
                f"Package 'segmentation-models-pytorch' is required to construct {normalized_name}. "
                "Install with: pip install segmentation-models-pytorch"
            )

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
        elif normalized_name == "fpn":
            return smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes
            )

    elif normalized_name == "segformer":
        if HAS_TRANSFORMERS:
            hf_encoder = "nvidia/mit-b0" if encoder_name in ["resnet34", "nvidia/mit-b0", "mit_b0", "mit-b0"] else encoder_name
            return HuggingFaceSegFormerWrapper(
                encoder_name=hf_encoder,
                in_channels=in_channels,
                num_classes=num_classes,
                encoder_weights=encoder_weights
            )
        elif HAS_SMP and hasattr(smp, "Segformer"):
            enc = "mit_b0" if encoder_name in ["resnet34", "nvidia/mit-b0", "mit_b0", "mit-b0"] else encoder_name
            return smp.Segformer(
                encoder_name=enc,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes
            )
        else:
            if allow_dummy_fallback:
                return FallbackUNetPlusPlus(in_channels=in_channels, num_classes=num_classes)
            raise ImportError(
                "Neither 'transformers' nor 'segmentation-models-pytorch' with Segformer support is available. "
                "Install with: pip install transformers"
            )

    raise ValueError(f"Unsupported model architecture: '{model_name}'")
