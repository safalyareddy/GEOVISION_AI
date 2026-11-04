#!/usr/bin/env python3
"""
GeoVision AI — Targeted Error Analysis & Diagnostic Visualizer
Generates 5-panel forensic comparison images:
1. Contrast-enhanced RGB Image
2. Ground Truth Mask
3. Predicted Sigmoid Probability Heatmap
4. Binary Prediction (tau=0.50)
5. Color-coded Error Map (Green=TP, Red=FP, Blue=FN, Gray=TN)
Analyzes coastal water, turbid estuaries, and narrow streams on held-out test data.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import build_model


def make_error_map(pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    """
    Constructs an RGB color-coded error map:
    - Green (0, 220, 0): True Positive
    - Red (230, 20, 20): False Positive
    - Blue (20, 100, 240): False Negative
    - Dark Gray (40, 40, 40): True Negative
    """
    h, w = pred_bin.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = [35, 35, 35] # Background land (TN)

    tp = (pred_bin == 1) & (gt_bin == 1)
    fp = (pred_bin == 1) & (gt_bin == 0)
    fn = (pred_bin == 0) & (gt_bin == 1)

    canvas[tp] = [0, 210, 80]    # Green: TP
    canvas[fp] = [230, 30, 30]   # Red: FP
    canvas[fn] = [30, 120, 255]  # Blue: FN
    return canvas


def generate_diagnostics_for_patches(patches: list, model_name: str, ckpt_path: str, out_dir: Path):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {model_name} from {ckpt_path} on {device}...")

    model = build_model(model_name=model_name, encoder_weights=None).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)

    for patch_stem in patches:
        img_path = Path("dataset/test/images") / f"{patch_stem}.png"
        msk_path = Path("dataset/test/masks") / f"{patch_stem}.png"

        if not img_path.exists() or not msk_path.exists():
            print(f"Skipping {patch_stem}: file not found")
            continue

        with Image.open(img_path) as im:
            raw_rgb = np.array(im.convert("RGB"))
        with Image.open(msk_path) as mk:
            gt_mask = (np.array(mk.convert("L")) >= 128).astype(np.uint8)

        # Contrast stretched RGB for visual inspection (satellite image is dark uint8)
        p2, p98 = np.percentile(raw_rgb, (2, 98))
        stretched_rgb = np.clip((raw_rgb - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)

        # PyTorch Tensor inference
        img_tensor = torch.from_numpy(raw_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        pred_bin = (probs >= 0.50).astype(np.uint8)
        error_map = make_error_map(pred_bin, gt_mask)

        # Metric computation for this single patch
        tp = int(np.sum((pred_bin == 1) & (gt_mask == 1)))
        fp = int(np.sum((pred_bin == 1) & (gt_mask == 0)))
        fn = int(np.sum((pred_bin == 0) & (gt_mask == 1)))
        dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
        iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
        prec = (tp + 1e-6) / (tp + fp + 1e-6)
        rec = (tp + 1e-6) / (tp + fn + 1e-6)

        # Generate 5-panel plot
        fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), dpi=150)
        fig.suptitle(
            f"Model: {model_name.upper()} | Test Patch: {patch_stem} | Dice: {dice:.3f} | IoU: {iou:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f}",
            fontsize=13, fontweight="bold", y=0.98
        )

        axes[0].imshow(stretched_rgb)
        axes[0].set_title("Stretched RGB Input", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="Blues_r")
        axes[1].set_title(f"Ground Truth ({np.sum(gt_mask):,} px)", fontsize=10)
        axes[1].axis("off")

        im_p = axes[2].imshow(probs, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[2].set_title(f"Sigmoid Probability\n(Mean: {probs.mean():.3f})", fontsize=10)
        axes[2].axis("off")
        plt.colorbar(im_p, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(pred_bin, cmap="Blues_r")
        axes[3].set_title(f"Prediction (th=0.50)\n({np.sum(pred_bin):,} px)", fontsize=10)
        axes[3].axis("off")

        axes[4].imshow(error_map)
        axes[4].set_title("Error Breakdown\nGreen=TP, Red=FP, Blue=FN", fontsize=10)
        axes[4].axis("off")

        plt.tight_layout()
        save_file = out_dir / f"diagnostic_{model_name}_{patch_stem}.png"
        plt.savefig(save_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_file.name} | Dice: {dice:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f}")


def main():
    diagnostic_patches = [
        "patch_0044",  # Large Coastal Bay (SegFormer Success)
        "patch_0004",  # Broad Coastal Inlet (DeepLabV3+ Success)
        "patch_0008",  # Tributary / Inland Transition
        "patch_0036",  # Turbid Estuary / Sediment (Hard Failure)
        "patch_0038",  # Intertidal Mudflat (Hard Failure)
        "patch_0039",  # Shallow Sediment Delta (Hard Failure)
        "patch_0045"   # Narrow Fragmented Stream (Resolution Failure)
    ]

    out_base = Path("reports/round2/visualizations")

    # Run for SegFormer (R2 Best Generalizer)
    generate_diagnostics_for_patches(
        diagnostic_patches,
        model_name="segformer",
        ckpt_path="checkpoints/experiments/round2/segformer_r2.pth",
        out_dir=out_base / "segformer"
    )

    # Run for DeepLabV3+ (R2 Best Conv Generalizer)
    generate_diagnostics_for_patches(
        diagnostic_patches,
        model_name="deeplabv3_plus",
        ckpt_path="checkpoints/experiments/round2/deeplabv3_plus_r2.pth",
        out_dir=out_base / "deeplabv3_plus"
    )

    # Run for UNet++ (R2 Inland Winner)
    generate_diagnostics_for_patches(
        diagnostic_patches,
        model_name="unet_plusplus",
        ckpt_path="checkpoints/experiments/round2/unet_plusplus_r2.pth",
        out_dir=out_base / "unet_plusplus"
    )

    print("\nVisual forensic diagnostics complete for all 3 models!")


if __name__ == "__main__":
    main()
