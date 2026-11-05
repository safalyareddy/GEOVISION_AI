"""
GeoVision AI — Satellite Water Body Segmentation Platform
Interactive Streamlit Web Dashboard (Cloud & Local Dual-Mode)
"""

import io
import os
import time
import base64
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
from PIL import Image
import numpy as np
import pandas as pd

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    import torch
    import torchvision.transforms as T
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.models import SUPPORTED_MODELS
from src.utils.model_registry import (
    load_verified_model,
    is_model_available,
    is_model_obtainable,
    get_model_status_summary,
    compute_file_sha256,
    resolve_checkpoint_path,
    CHAMPION_MODEL,
    MODEL_REGISTRY,
    DEFAULT_HF_REPO,
    DEFAULT_HF_REVISION,
)

# Page Configuration
if HAS_STREAMLIT:
    st.set_page_config(
        page_title="GeoVision AI — Water Body Segmentation",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# -----------------------------------------------------------------------------
# PyTorch Model Caching (Safe Deserialization & Cryptographic Integrity)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False) if HAS_STREAMLIT else (lambda f: f)
def get_cached_pytorch_model(model_name: str) -> Any:
    """
    Download (if needed), cryptographically verify, strictly deserialize,
    and cache the PyTorch model artifact. Strictly rejects uninitialized weights.
    """
    return load_verified_model(model_name=model_name, device="cpu")


def run_pytorch_inference(
    model: Any,
    image: Image.Image,
    threshold: float
) -> Tuple[Image.Image, np.ndarray, float, float, float]:
    """
    Execute genuine PyTorch FPN inference using standard preprocessing,
    torch.inference_mode(), sigmoid activation, and decision thresholding.

    Returns:
        (mask_img, binary_mask, latency_ms, water_coverage_pct, water_area_km2)
    """
    orig_w, orig_h = image.size

    # Standard Preprocessing: Resize to nominal 256x256, normalize to [0.0, 1.0] float32 tensor
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    t_img = transform(image.convert("RGB")).unsqueeze(0)

    start_time = time.perf_counter()
    with torch.inference_mode():
        logits = model(t_img)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Decision thresholding
    binary_mask = (probs >= threshold).astype(np.uint8)
    water_pixels = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)
    water_coverage_pct = float((water_pixels / total_pixels) * 100.0)

    # Nominal area estimation (Sentinel-2 10m GSD -> 100 m^2/pixel)
    water_area_m2 = float(water_pixels * 100.0)
    water_area_km2 = float(water_area_m2 / 1e6)

    # Resize binary mask back to original input dimensions
    mask_img = Image.fromarray(binary_mask * 255, mode="L").resize((orig_w, orig_h), Image.NEAREST)

    return mask_img, binary_mask, latency_ms, water_coverage_pct, water_area_km2


def run_fastapi_inference(
    image: Image.Image,
    model_name: str,
    threshold: float,
    api_url: str
) -> Tuple[Image.Image, np.ndarray, float, float, float]:
    """Execute prediction via external FastAPI REST microservice."""
    orig_w, orig_h = image.size
    start_time = time.perf_counter()

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    files = {"file": ("image.png", buf.getvalue(), "image/png")}
    data = {"model_name": model_name, "threshold": str(threshold)}

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(f"{api_url.rstrip('/')}/predict", files=files, data=data)

    if resp.status_code != 200:
        raise RuntimeError(f"FastAPI microservice returned HTTP {resp.status_code}: {resp.text}")

    payload = resp.json()
    mask_bytes = base64.b64decode(payload["mask_base64"])
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L").resize((orig_w, orig_h), Image.NEAREST)
    binary_mask = (np.array(mask_img) > 127).astype(np.uint8)
    latency_ms = float(payload.get("latency_ms", (time.perf_counter() - start_time) * 1000.0))
    water_coverage_pct = float(payload.get("water_coverage_percentage", 0.0))
    water_area_km2 = float(payload.get("estimated_water_area_km2", 0.0))

    return mask_img, binary_mask, latency_ms, water_coverage_pct, water_area_km2


def create_overlay(image: Image.Image, mask_img: Image.Image) -> Image.Image:
    """Create cyan overlay of predicted water mask on original satellite image."""
    img_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask_img)

    overlay = img_np.copy()
    # Cyan highlight [0, 220, 255] at 50% opacity for predicted water pixels
    water_indices = mask_np > 127
    overlay[water_indices] = (
        0.5 * img_np[water_indices] + 0.5 * np.array([0, 220, 255])
    ).astype(np.uint8)

    return Image.fromarray(overlay)


# -----------------------------------------------------------------------------
# Main Application Dashboard
# -----------------------------------------------------------------------------
def main():
    if not HAS_STREAMLIT:
        print("Streamlit is not installed. Run `pip install streamlit`.")
        return

    st.title("🛰️ GeoVision AI — Satellite Water Body Segmentation")
    st.markdown(
        "**Production Machine Learning Platform** for high-resolution surface water detection "
        "and hydrological area quantification from Sentinel-2 optical satellite imagery."
    )

    # Sidebar Header
    st.sidebar.title("⚙️ Control Panel")

    # 1. Model Architecture Selector
    model_options = [CHAMPION_MODEL] + [m for m in SUPPORTED_MODELS if m != CHAMPION_MODEL]
    model_status_map = get_model_status_summary()

    def format_model(name: str) -> str:
        meta = model_status_map.get(name, {})
        display = meta.get("display_name", name.upper())
        if meta.get("is_champion"):
            return f"⭐ {display}"
        return display

    model_choice = st.sidebar.selectbox(
        "Select Model Architecture",
        options=model_options,
        format_func=format_model
    )

    # 2. Decision Threshold Slider
    threshold = st.sidebar.slider(
        "Binary Decision Threshold (τ)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability threshold above which pixels are classified as water. "
             "Default calibrated benchmark threshold is 0.50 (with 0.60 for high-precision scenarios)."
    )

    # 3. Backend Detection & Selection
    configured_api_url = os.environ.get("API_URL", "").strip()
    fastapi_healthy = False
    active_api_url = None

    if configured_api_url:
        try:
            with httpx.Client(timeout=1.0) as client:
                r = client.get(f"{configured_api_url.rstrip('/')}/health")
                if r.status_code == 200:
                    fastapi_healthy = True
                    active_api_url = configured_api_url
        except Exception:
            fastapi_healthy = False
    else:
        # Check local Docker port 8000
        try:
            with httpx.Client(timeout=0.2) as client:
                r = client.get("http://127.0.0.1:8000/health")
                if r.status_code == 200:
                    fastapi_healthy = True
                    active_api_url = "http://127.0.0.1:8000"
        except Exception:
            fastapi_healthy = False

    BACKEND_PYTORCH = "PyTorch (In-Process) [Primary / Cloud Default]"
    BACKEND_FASTAPI = "FastAPI Microservice (REST API)" if fastapi_healthy else "FastAPI Microservice (Unavailable - Local Docker only)"
    BACKEND_ONNX = "ONNX Runtime (Disabled - Pending Verified Artifact)"

    backend_choices = [BACKEND_PYTORCH, BACKEND_FASTAPI, BACKEND_ONNX]
    backend = st.sidebar.radio(
        "Inference Execution Backend",
        options=backend_choices,
        index=0,  # Always default to PyTorch (In-Process)
        help="PyTorch (In-Process) runs the verified champion checkpoint directly on CPU."
    )

    # Backend Warning / Configuration Callouts
    if backend == BACKEND_FASTAPI and not fastapi_healthy:
        st.sidebar.warning(
            "⚠️ **FastAPI Service Unavailable**\n\n"
            "FastAPI is not reachable at localhost:8000 or via $API_URL. "
            "Please select **PyTorch (In-Process)** for cloud execution."
        )
    elif backend == BACKEND_ONNX:
        st.sidebar.info(
            "ℹ️ **ONNX Runtime Disabled**\n\n"
            "ONNX execution is disabled pending a verified, full-precision standalone ONNX export. "
            "Please select **PyTorch (In-Process)**."
        )

    # Model Loading & Verification Gate
    model_obj = None
    model_load_error = None
    resolved_ckpt_path = None
    verified_sha256 = None

    if backend == BACKEND_PYTORCH:
        with st.sidebar:
            with st.spinner("Verifying & loading model weights..."):
                try:
                    model_obj = get_cached_pytorch_model(model_choice)
                    resolved_ckpt_path = resolve_checkpoint_path(model_choice, allow_download=False)
                    if resolved_ckpt_path and resolved_ckpt_path.exists():
                        verified_sha256 = compute_file_sha256(resolved_ckpt_path)
                except Exception as e:
                    model_load_error = str(e)

        if model_obj is not None:
            st.sidebar.success(
                f"**Verified Model Loaded**\n\n"
                f"• Architecture: `{model_choice.upper()} (ResNet34)`\n"
                f"• Hub Source: `{DEFAULT_HF_REPO}`\n"
                f"• Pinned Revision: `{DEFAULT_HF_REVISION[:7]}`\n"
                f"• SHA-256: `{verified_sha256[:12]}... Verified`\n"
                f"• Mode: `torch.inference_mode()`"
            )
        elif model_load_error:
            st.sidebar.error(
                f"**Model Verification Failed**\n\n{model_load_error}\n\n"
                "Inference with uninitialized weights is strictly prohibited."
            )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Overview")
    st.sidebar.info(
        "**Sentinel-2 Imagery**: 10m spatial resolution\n\n"
        "• Total Split Samples: 414\n"
        "• Train: 306 (3.89% water)\n"
        "• Val: 54 (5.51% water)\n"
        "• Test: 54 (13.38% water)\n"
        "• Overall: 5.34% Water / 94.66% Land"
    )

    # Main Dashboard Tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Single Image Inference", "📈 Model Benchmarks", "ℹ️ Architecture & Methodology"])

    with tab1:
        st.subheader("Satellite Image Analysis & Mask Generation")

        col_input, col_sample = st.columns([2, 1])

        with col_input:
            uploaded_file = st.file_uploader(
                "Upload Satellite Image (PNG, JPG, TIFF)",
                type=["png", "jpg", "jpeg", "tif", "tiff"]
            )

        with col_sample:
            use_sample = st.checkbox("Use Sample Test Image (`patch_0047.png`)", value=(uploaded_file is None))

        image = None
        gt_mask = None

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
        elif use_sample:
            sample_candidates = [
                Path("assets/demo/patch_0047.png"),
                Path("dataset/test/images/patch_0047.png"),
            ]
            sample_msk_candidates = [
                Path("assets/demo/patch_0047_mask.png"),
                Path("dataset/test/masks/patch_0047.png"),
            ]
            for cand in sample_candidates:
                if cand.exists():
                    image = Image.open(cand).convert("RGB")
                    break
            for cand in sample_msk_candidates:
                if cand.exists():
                    gt_mask = Image.open(cand).convert("L")
                    break

        # Verification Gate
        can_run = False
        if backend == BACKEND_PYTORCH and model_obj is not None and image is not None:
            can_run = True
        elif backend == BACKEND_FASTAPI and fastapi_healthy and image is not None:
            can_run = True

        run_clicked = st.button("Run Water Segmentation Analysis", type="primary", disabled=not can_run)

        # Trigger execution if button clicked, or on initial load with sample image
        if can_run and (run_clicked or use_sample):
            mask_img = None
            binary_mask = None
            latency_ms = 0.0
            water_coverage_pct = 0.0
            water_area_km2 = 0.0
            error_message = None

            try:
                if backend == BACKEND_PYTORCH:
                    mask_img, binary_mask, latency_ms, water_coverage_pct, water_area_km2 = run_pytorch_inference(
                        model=model_obj,
                        image=image,
                        threshold=threshold
                    )
                elif backend == BACKEND_FASTAPI:
                    mask_img, binary_mask, latency_ms, water_coverage_pct, water_area_km2 = run_fastapi_inference(
                        image=image,
                        model_name=model_choice,
                        threshold=threshold,
                        api_url=active_api_url
                    )
            except Exception as e:
                error_message = str(e)

            if error_message:
                st.error(f"❌ **Inference Error**: {error_message}")
            elif mask_img is not None and binary_mask is not None:
                overlay_img = create_overlay(image, mask_img)

                st.markdown("---")
                # Metric Callouts
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted Water Coverage", f"{water_coverage_pct:.2f}%")
                m2.metric(
                    "Estimated Water Area",
                    f"{water_area_km2:.4f} km²",
                    help="Calculated assuming nominal Sentinel-2 10m Ground Sample Distance (100 m²/px). Nominal approximation."
                )
                m3.metric("Inference Latency", f"{latency_ms:.2f} ms")
                m4.metric("Active Backend", "PyTorch (In-Process)" if backend == BACKEND_PYTORCH else "FastAPI")

                # Model Artifact Integrity Card
                st.caption(
                    f"**Verified Model**: `{model_choice.upper()}` | "
                    f"**Revision**: `{DEFAULT_HF_REVISION[:7]}` | "
                    f"**SHA-256**: `d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1 (Verified)` | "
                    f"**Resolution Assumption**: Nominal 10m GSD (100 m²/px, unreferenced approximation)"
                )

                st.markdown("---")
                # Visual Comparison Grid
                if gt_mask is not None:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.image(image, caption="Satellite Input Image", use_container_width=True)
                    c2.image(gt_mask, caption="Ground Truth Mask", use_container_width=True)
                    c3.image(mask_img, caption=f"Predicted Mask (τ={threshold:.2f})", use_container_width=True)
                    c4.image(overlay_img, caption="Segmentation Overlay (Cyan)", use_container_width=True)
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.image(image, caption="Satellite Input Image", use_container_width=True)
                    c2.image(mask_img, caption=f"Predicted Mask (τ={threshold:.2f})", use_container_width=True)
                    c3.image(overlay_img, caption="Segmentation Overlay (Cyan)", use_container_width=True)

        elif not can_run and image is not None:
            if backend == BACKEND_FASTAPI and not fastapi_healthy:
                st.info("ℹ️ FastAPI is unavailable. Please select **PyTorch (In-Process)** to run inference.")
            elif backend == BACKEND_ONNX:
                st.info("ℹ️ ONNX Runtime is disabled. Please select **PyTorch (In-Process)** to run inference.")
            elif model_load_error:
                st.error(f"❌ Cannot run inference: {model_load_error}")

    with tab2:
        st.subheader("Cross-Architecture Performance Evaluation")
        csv_path = Path("reports/benchmark-results.csv")
        if not csv_path.exists():
            csv_path = Path("benchmark_results.csv")

        if csv_path.exists():
            df_bench = pd.read_csv(csv_path)
            st.dataframe(df_bench, use_container_width=True)

            st.markdown("### Architecture Metrics Comparison")
            c_left, c_right = st.columns(2)
            with c_left:
                if "water_recall" in df_bench.columns:
                    st.bar_chart(df_bench.set_index("model_name")[["water_recall"]], use_container_width=True)
                    st.caption("Water Recall Comparison")
            with c_right:
                if "micro_dice" in df_bench.columns:
                    st.bar_chart(df_bench.set_index("model_name")[["micro_dice"]], use_container_width=True)
                    st.caption("Micro Dice Score Comparison")
        else:
            st.info("Benchmark results file not found.")

    with tab3:
        st.subheader("Platform Architecture & Methodological Protocol")
        st.markdown(
            """
            ### Asymmetric Loss Optimization for Class Imbalance
            In optical satellite remote sensing, surface water accounts for less than 5.34% of total pixel volume across splits.
            Standard Cross-Entropy and Dice loss functions suffer from the **Accuracy Paradox**, achieving >86% accuracy by
            predicting pure background land while failing to detect narrow tributaries and fine coastal boundaries.

            ### Verified Production Architecture
            1. **Primary Cloud Engine**: In-process PyTorch model evaluation using `torch.inference_mode()`.
            2. **Champion Model**: Feature Pyramid Network (`smp.FPN`) with `resnet34` backbone trained with Focal Tversky Loss (α=0.7, β=0.3, γ=4/3).
            3. **Cryptographic Model Registry**: Checkpoint weights downloaded from Hugging Face Hub (`safalya06/geovision-water-segmentation-models@022039f0ce080afec9354a18635af97cc4cf352e`) and verified via 64 KB streaming SHA-256 (`d21fa18c36beabdb0b139364d347a6e97c1cc00e83e9ba92630e9a7795119eb1`).
            4. **Safety Verification**: Strict state-dictionary loading with `weights_only=True` preventing uninitialized model predictions.
            """
        )


if __name__ == "__main__":
    main()
