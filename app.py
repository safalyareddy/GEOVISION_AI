"""
GeoVision AI — Satellite Water Body Segmentation Platform
Interactive Streamlit Web Dashboard
"""

import io
import os
import time
import base64
from pathlib import Path
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

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from src.models import build_model, SUPPORTED_MODELS
from src.utils.model_registry import (
    load_verified_model,
    is_model_available,
    get_model_status_summary,
    CHAMPION_MODEL
)

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Page Configuration
if HAS_STREAMLIT:
    st.set_page_config(
        page_title="GeoVision AI — Water Body Segmentation",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Model Caching Helpers
@st.cache_resource if HAS_STREAMLIT else (lambda f: f)
def load_pytorch_model(model_name: str):
    """Load and cache verified PyTorch segmentation model, strictly preventing uninitialized weights."""
    return load_verified_model(model_name=model_name, device="cpu")


@st.cache_resource if HAS_STREAMLIT else (lambda f: f)
def load_onnx_session(model_name: str):
    """Load and cache ONNX Runtime inference session."""
    onnx_map = {
        "unet_plusplus": "checkpoints/unet_plusplus_best.onnx",
        "deeplabv3_plus": "checkpoints/deeplabv3_plus_best.onnx",
        "segformer": "checkpoints/segformer_best.onnx"
    }
    onnx_path = Path(onnx_map.get(model_name, ""))
    if onnx_path.exists() and HAS_ONNX:
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return None


def run_inference(image: Image.Image, model_name: str, threshold: float, backend: str, api_url: str = DEFAULT_API_URL):
    """Run model inference and return binary mask, probabilities, latency."""
    orig_size = image.size

    # 1. FastAPI Microservice Execution
    if "FastAPI" in backend and HAS_HTTPX:
        start_time = time.perf_counter()
        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            files = {"file": ("image.png", buf.getvalue(), "image/png")}
            data = {"model_name": model_name, "threshold": str(threshold)}
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(f"{api_url.rstrip('/')}/predict", files=files, data=data)
            if resp.status_code == 200:
                payload = resp.json()
                mask_bytes = base64.b64decode(payload["mask_base64"])
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L").resize(orig_size, Image.NEAREST)
                binary_mask = (np.array(mask_img) > 127).astype(np.uint8)
                latency_ms = payload.get("latency_ms", (time.perf_counter() - start_time) * 1000.0)
                return mask_img, binary_mask, latency_ms
            else:
                if HAS_STREAMLIT:
                    st.error(f"FastAPI error ({resp.status_code}): {resp.text}. Falling back to in-process execution.")
        except Exception as e:
            if HAS_STREAMLIT:
                st.warning(f"Could not connect to FastAPI service at '{api_url}': {e}. Falling back to local execution.")

    # 2. Local In-Process Execution (ONNX Runtime or PyTorch)
    img_resized = image.resize((256, 256))
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    start_time = time.perf_counter()

    if "ONNX" in backend and HAS_ONNX:
        session = load_onnx_session(model_name)
        if session is not None:
            input_tensor = np.transpose(img_arr, (2, 0, 1))[np.newaxis, :].astype(np.float32)
            input_name = session.get_inputs()[0].name
            logits = session.run(None, {input_name: input_tensor})[0]
            probs = 1.0 / (1.0 + np.exp(-logits.squeeze()))
        else:
            session = None

    if "ONNX" not in backend or session is None:
        model = load_pytorch_model(model_name)
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        t_img = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(t_img)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    binary_mask = (probs >= threshold).astype(np.uint8)
    mask_img = Image.fromarray(binary_mask * 255, mode="L").resize(orig_size, Image.NEAREST)

    return mask_img, binary_mask, latency_ms


def create_overlay(image: Image.Image, mask_img: Image.Image):
    """Create cyan overlay of predicted water mask on original satellite image."""
    img_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask_img)

    overlay = img_np.copy()
    # Cyan highlight for predicted water body pixels [0, 255, 255]
    water_indices = mask_np > 127
    overlay[water_indices] = (
        0.5 * img_np[water_indices] + 0.5 * np.array([0, 220, 255])
    ).astype(np.uint8)

    return Image.fromarray(overlay)


def main():
    if not HAS_STREAMLIT:
        print("Streamlit is not installed. Run `pip install streamlit`.")
        return

    st.title("🌊 GeoVision AI — Satellite Water Body Segmentation Platform")
    st.markdown(
        "**MAANG-Grade Machine Learning & Remote Sensing Platform** for high-resolution "
        "satellite surface water detection, water coverage quantization, and cross-architecture benchmarking."
    )

    # Sidebar
    st.sidebar.header("⚙️ Inference Configuration")

    model_status = get_model_status_summary()
    model_options = [CHAMPION_MODEL] + [m for m in ["unet_plusplus", "deeplabv3_plus", "segformer"] if m != CHAMPION_MODEL]

    def format_model(name: str) -> str:
        info = model_status.get(name, {})
        is_champ = info.get("is_champion", False)
        avail = info.get("available", False)
        names = {
            "fpn": "Feature Pyramid Network (FPN)",
            "unet_plusplus": "UNet++ (ResNet-34)",
            "deeplabv3_plus": "DeepLabV3+ (ResNet-34)",
            "segformer": "SegFormer-B0 (Transformer)"
        }
        base = names.get(name, name)
        if is_champ and avail:
            return f"{base} — 🏆 Champion (Weights Ready)"
        elif avail:
            return f"{base} — (Weights Ready)"
        else:
            return f"{base} — (Weights Missing)"

    model_choice = st.sidebar.selectbox(
        "Select Model Architecture",
        options=model_options,
        format_func=format_model
    )

    threshold = st.sidebar.slider(
        "Binary Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability threshold above which a pixel is classified as surface water. Benchmark calibrated threshold is 0.50 (with 0.60 for high-precision scenarios)."
    )

    backend = st.sidebar.radio(
        "Inference Execution Backend",
        options=["FastAPI Microservice (REST API)", "PyTorch (In-Process)", "ONNX Runtime (In-Process)"],
        help="Compare execution performance between remote FastAPI microservice, PyTorch eager execution, and ONNX Runtime."
    )

    api_url = DEFAULT_API_URL
    if "FastAPI" in backend:
        api_url = st.sidebar.text_input("FastAPI Endpoint URL", value=DEFAULT_API_URL, help="Base URL for FastAPI microservice (e.g. http://localhost:8000 or http://api:8000)")

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
            uploaded_file = st.file_uploader("Upload Satellite Image (PNG, JPG, TIFF)", type=["png", "jpg", "jpeg", "tif", "tiff"])

        with col_sample:
            use_sample = st.checkbox("Use Sample Test Image (`patch_0047.png`)", value=(uploaded_file is None))

        image = None
        gt_mask = None

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
        elif use_sample:
            sample_img_path = Path("dataset/test/images/patch_0047.png")
            sample_msk_path = Path("dataset/test/masks/patch_0047.png")
            if sample_img_path.exists():
                image = Image.open(sample_img_path).convert("RGB")
            if sample_msk_path.exists():
                gt_mask = Image.open(sample_msk_path).convert("L")

        model_ready = is_model_available(model_choice)
        if not model_ready and "FastAPI" not in backend:
            st.warning(
                f"⚠️ Trained weights for **{model_choice}** are not loaded locally. "
                "Inference with uninitialized weights is strictly prevented to avoid invalid predictions. "
                "Please select the Champion FPN model or provide verified weights."
            )

        if image is not None and (model_ready or "FastAPI" in backend):
            try:
                mask_img, binary_mask, latency_ms = run_inference(image, model_choice, threshold, backend, api_url=api_url)
            except Exception as e:
                st.error(f"Inference execution failed: {e}")
                mask_img = None

            if mask_img is not None:
                overlay_img = create_overlay(image, mask_img)

                # Calculate water coverage
                water_pixels = int(np.sum(binary_mask > 0))
                total_pixels = int(binary_mask.size)
                water_coverage_pct = (water_pixels / total_pixels) * 100.0

                # Estimate physical area (10m x 10m per pixel for Sentinel-2 = 100 m^2)
                water_area_km2 = (water_pixels * 100) / 1e6

                st.markdown("---")
                # Metric Callouts
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted Water Coverage", f"{water_coverage_pct:.2f}%")
                m2.metric(
                    "Estimated Water Area",
                    f"{water_area_km2:.3f} km²",
                    help="Nominal physical area assuming Sentinel-2 10m GSD (100 m²/px). Unreferenced approximation when geospatial CRS/GeoTIFF metadata is not present."
                )
                m3.metric("Inference Latency", f"{latency_ms:.2f} ms")
                m4.metric("Active Backend", backend)

                st.markdown("---")
                # Visual Grid
                if gt_mask is not None:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.image(image, caption="Satellite Input Image", use_column_width=True)
                    c2.image(gt_mask, caption="Ground Truth Mask", use_column_width=True)
                    c3.image(mask_img, caption=f"Predicted Mask (Thresh={threshold})", use_column_width=True)
                    c4.image(overlay_img, caption="Segmentation Overlay", use_column_width=True)
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.image(image, caption="Satellite Input Image", use_column_width=True)
                    c2.image(mask_img, caption=f"Predicted Mask (Thresh={threshold})", use_column_width=True)
                    c3.image(overlay_img, caption="Segmentation Overlay", use_column_width=True)

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
            st.warning("Benchmark results file not found.")

    with tab3:
        st.subheader("Technical Architecture & System Methodology")
        st.markdown("""
        ### Architectural Breakdown

        1. **Model Backbone Selection**:
           - **Feature Pyramid Network (FPN)**: Top-down feature hierarchy combining semantic and high-resolution spatial features for thin channels and fine shorelines (Champion Model).
           - **UNet++**: Nested dense skip pathways reduce semantic gap between feature maps.
           - **DeepLabV3+**: Atrous Spatial Pyramid Pooling (ASPP) captures multi-scale context.
           - **SegFormer**: Lightweight Transformer encoder with MLP decoder.

        2. **Loss Function (Recall Prioritization)**:
           - Asymmetric **Tversky and Focal Tversky Loss** ($\beta=0.7 > \alpha=0.3$) heavily penalizes false negatives.
           - Designed to counteract extreme class imbalance (94.66% land vs 5.34% surface water overall; 3.89% in training set).

        3. **Water Coverage Computation**:
           $$\\text{Water Coverage (\\%)} = \\frac{\\text{Count}(\\text{Pixel}_{\\text{pred}} \\ge \\tau)}{\\text{Total Valid Pixels}} \\times 100$$
           *Note*: Water coverage represents physical surface area fraction, distinct from model accuracy or Dice score.
        """)


if __name__ == "__main__":
    main()
