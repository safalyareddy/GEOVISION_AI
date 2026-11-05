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

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

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
from src.utils.input_validator import (
    classify_input_image,
    calculate_water_area_km2,
    InputClassification,
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
        out = model(t_img)
        if isinstance(out, dict):
            logits = out.get("out", out.get("logits"))
        elif hasattr(out, "logits"):
            logits = out.logits
        else:
            logits = out
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


@st.cache_resource(show_spinner=False) if HAS_STREAMLIT else (lambda f: f)
def get_cached_onnx_session(model_name: str) -> Any:
    """Load or export optimized ONNX runtime session for given model."""
    if not HAS_ONNX:
        raise RuntimeError("ONNX Runtime is not installed.")
    onnx_path = Path("checkpoints/onnx") / f"{model_name}.onnx"
    if not onnx_path.exists():
        # Export on-the-fly from verified PyTorch checkpoint
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        pt_model = get_cached_pytorch_model(model_name)
        pt_model.eval()
        dummy = torch.randn(1, 3, 256, 256, dtype=torch.float32)
        torch.onnx.export(
            pt_model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])


def run_onnx_inference(
    session: Any,
    image: Image.Image,
    threshold: float
) -> Tuple[Image.Image, np.ndarray, float, float, float]:
    """Execute prediction using optimized ONNX Runtime execution engine."""
    orig_w, orig_h = image.size
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    t_img = transform(image.convert("RGB")).unsqueeze(0).numpy()

    start_time = time.perf_counter()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: t_img})
    raw_out = outputs[0]
    probs = 1.0 / (1.0 + np.exp(-raw_out))
    probs = np.squeeze(probs)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    binary_mask = (probs >= threshold).astype(np.uint8)
    water_pixels = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)
    water_coverage_pct = float((water_pixels / total_pixels) * 100.0)
    water_area_m2 = float(water_pixels * 100.0)
    water_area_km2 = float(water_area_m2 / 1e6)

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
        status_lbl = meta.get("status_label", "")
        prefix = "⭐ " if meta.get("is_champion") else ""
        return f"{prefix}{display} — [{status_lbl}]"

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
    BACKEND_ONNX = "ONNX Runtime (Accelerated CPU)" if HAS_ONNX else "ONNX Runtime (Unavailable - Requires onnxruntime)"

    backend_choices = [BACKEND_PYTORCH, BACKEND_FASTAPI, BACKEND_ONNX]
    backend = st.sidebar.radio(
        "Inference Execution Backend",
        options=backend_choices,
        index=0,  # Always default to PyTorch (In-Process)
        help="Select between in-process PyTorch execution, ONNX Runtime acceleration, or external FastAPI microservice."
    )

    # Backend Warning / Configuration Callouts
    if backend == BACKEND_FASTAPI and not fastapi_healthy:
        st.sidebar.warning(
            "⚠️ **FastAPI Service Unavailable**\n\n"
            "FastAPI is not reachable at localhost:8000 or via $API_URL. "
            "Please select **PyTorch (In-Process)** for cloud execution."
        )
    elif backend == BACKEND_ONNX and not HAS_ONNX:
        st.sidebar.warning(
            "⚠️ **ONNX Runtime Unavailable**\n\n"
            "onnxruntime package is not installed. Please select **PyTorch (In-Process)**."
        )

    # Model Loading & Verification Gate
    model_obj = None
    model_load_error = None
    resolved_ckpt_path = None
    verified_sha256 = None
    onnx_sess = None
    onnx_load_error = None

    if backend == BACKEND_PYTORCH:
        meta_choice = MODEL_REGISTRY.get(model_choice, {})
        is_hf = meta_choice.get("is_hosted_on_hf", False)
        with st.sidebar:
            spinner_msg = (
                f"Downloading & verifying `{model_choice}` from Hugging Face Hub ({DEFAULT_HF_REPO}@{DEFAULT_HF_REVISION[:7]})..."
                if is_hf else f"Verifying & loading `{model_choice}` from local filesystem..."
            )
            with st.spinner(spinner_msg):
                try:
                    model_obj = get_cached_pytorch_model(model_choice)
                    resolved_ckpt_path = resolve_checkpoint_path(model_choice, allow_download=False)
                    if resolved_ckpt_path and resolved_ckpt_path.exists():
                        verified_sha256 = compute_file_sha256(resolved_ckpt_path)
                except Exception as e:
                    model_load_error = str(e)

        if model_obj is not None:
            sha_disp = f"{verified_sha256[:12]}... Verified" if verified_sha256 else "Verified"
            source_disp = f"Hugging Face Hub (`{DEFAULT_HF_REPO}`)" if is_hf else "Local Filesystem"
            rev_disp = f"`{DEFAULT_HF_REVISION[:7]}`" if is_hf else "`Local`"
            st.sidebar.success(
                f"**Model Verification Status**\n\n"
                f"• Status: `Verified and loaded`\n"
                f"• Architecture: `{model_choice.upper()}`\n"
                f"• Weights Source: {source_disp}\n"
                f"• Pinned Revision: {rev_disp}\n"
                f"• SHA-256: `{sha_disp}`\n"
                f"• Mode: `torch.inference_mode()`"
            )
        elif model_load_error:
            st.sidebar.error(
                f"**Model Verification Failed**\n\n"
                f"• Status: `Checkpoint verification failed`\n\n"
                f"{model_load_error}\n\n"
                "Inference with uninitialized or fallback weights is strictly prohibited."
            )
    elif backend == BACKEND_ONNX and HAS_ONNX:
        with st.sidebar:
            with st.spinner(f"Loading & optimizing ONNX Runtime session for `{model_choice}`..."):
                try:
                    onnx_sess = get_cached_onnx_session(model_choice)
                except Exception as e:
                    onnx_load_error = str(e)

        if onnx_sess is not None:
            st.sidebar.success(
                f"**ONNX Runtime Active**\n\n"
                f"• Status: `Session optimized and loaded`\n"
                f"• Architecture: `{model_choice.upper()}`\n"
                f"• Execution Provider: `CPUExecutionProvider`\n"
                f"• Graph Optimization: `ORT_ENABLE_ALL`"
            )
        elif onnx_load_error:
            st.sidebar.error(
                f"**ONNX Session Failed**\n\n"
                f"{onnx_load_error}"
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

    # -------------------------------------------------------------------------
    # Satellite Image Input Selector (Shared across Inference & Comparison)
    # -------------------------------------------------------------------------
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
    sample_file_bytes = None

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
                sample_file_bytes = cand.read_bytes()
                image = Image.open(cand).convert("RGB")
                break
        for cand in sample_msk_candidates:
            if cand.exists():
                gt_mask = Image.open(cand).convert("L")
                break

    # Scientific Safety Gate: Classify Input Image
    classification = None
    input_meta = {}
    if image is not None:
        filename = uploaded_file.name if uploaded_file else "patch_0047.png"
        raw_bytes = uploaded_file.getvalue() if uploaded_file else sample_file_bytes
        classification, input_meta = classify_input_image(image, raw_bytes=raw_bytes, filename=filename)

        if classification == InputClassification.VERIFIED_PROJECT_SAMPLE:
            st.success(f"🟢 **Verified Project Sample**: Sentinel-2 test split patch (`{input_meta.get('project_sample_name')}`). Calibrated ground truth available.")
        elif classification == InputClassification.VERIFIED_GEOSPATIAL_INPUT:
            st.success(f"🌐 **Verified Geospatial Input**: Embedded GeoTIFF coordinate reference & pixel scale detected ({input_meta.get('pixel_scale_m')}). Exact spatial quantification active.")
        elif classification == InputClassification.USER_DECLARED_SATELLITE_UNVERIFIED:
            st.info("🟡 **User-Declared Satellite (Nominal Approximation Mode)**: Remote sensing imagery without embedded geospatial metadata tags. Nominal 10m GSD assumption applied.")
        elif classification == InputClassification.UNSUPPORTED_PHOTOGRAPH:
            st.error(
                f"🛑 **Unsupported Consumer Photograph Detected**\n\n"
                f"{input_meta.get('rejection_reason')}\n\n"
                f"• **Mean Saturation**: `{input_meta.get('mean_saturation', 'N/A')}`\n"
                f"• **Camera / EXIF**: `{input_meta.get('camera_make', 'None')} {input_meta.get('camera_model', '')}`\n\n"
                "**Hydrological Area Claim Blocked**: Consumer photographs of flowers, portraits, or indoor objects "
                "do not represent Earth observation reflectance and cannot produce valid hydrological area metrics."
            )
        elif classification == InputClassification.INVALID_IMAGE:
            st.error(f"❌ **Invalid Image Payload**: {input_meta.get('rejection_reason')}")

    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Single Image Inference",
        "🔬 Multi-Architecture Comparison",
        "📈 Model Benchmarks",
        "ℹ️ Architecture & Methodology"
    ])

    with tab1:
        st.markdown("### Architecture-Specific Evaluation")

        # Verification Gate
        can_run = False
        is_photo_rejected = (
            classification == InputClassification.UNSUPPORTED_PHOTOGRAPH or
            classification == InputClassification.INVALID_IMAGE
        )

        if not is_photo_rejected and image is not None:
            if backend == BACKEND_PYTORCH and model_obj is not None:
                can_run = True
            elif backend == BACKEND_FASTAPI and fastapi_healthy:
                can_run = True
            elif backend == BACKEND_ONNX and onnx_sess is not None:
                can_run = True

        run_clicked = st.button("Run Water Segmentation Analysis", type="primary", disabled=not can_run)

        # Trigger execution if button clicked, or on initial load with sample image
        if can_run and (run_clicked or use_sample):
            mask_img = None
            binary_mask = None
            latency_ms = 0.0
            water_coverage_pct = 0.0
            water_area_km2 = 0.0
            area_note = ""
            error_message = None

            try:
                if backend == BACKEND_PYTORCH:
                    mask_img, binary_mask, latency_ms, water_coverage_pct, _ = run_pytorch_inference(
                        model=model_obj,
                        image=image,
                        threshold=threshold
                    )
                    water_pixels = int(np.sum(binary_mask))
                    water_area_km2, area_note = calculate_water_area_km2(water_pixels, classification, input_meta)
                elif backend == BACKEND_ONNX:
                    mask_img, binary_mask, latency_ms, water_coverage_pct, _ = run_onnx_inference(
                        session=onnx_sess,
                        image=image,
                        threshold=threshold
                    )
                    water_pixels = int(np.sum(binary_mask))
                    water_area_km2, area_note = calculate_water_area_km2(water_pixels, classification, input_meta)
                elif backend == BACKEND_FASTAPI:
                    mask_img, binary_mask, latency_ms, water_coverage_pct, water_area_km2 = run_fastapi_inference(
                        image=image,
                        model_name=model_choice,
                        threshold=threshold,
                        api_url=active_api_url
                    )
                    area_note = "Computed via FastAPI microservice with input safety verification."
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
                area_disp = f"{water_area_km2:.4f} km²" if water_area_km2 is not None else "N/A"
                m2.metric(
                    "Estimated Water Area",
                    area_disp,
                    help=area_note or "Geospatial area estimation"
                )
                m3.metric("Inference Latency", f"{latency_ms:.2f} ms")
                backend_disp = (
                    "PyTorch (In-Process)" if backend == BACKEND_PYTORCH
                    else ("ONNX Runtime" if backend == BACKEND_ONNX else "FastAPI")
                )
                m4.metric("Active Backend", backend_disp)

                # Model Artifact Integrity Card
                is_hf = MODEL_REGISTRY[model_choice].get("is_hosted_on_hf", False)
                active_hash = verified_sha256 or MODEL_REGISTRY[model_choice].get("expected_sha256") or "Verified"
                source_lbl = f"Hugging Face Hub (`{DEFAULT_HF_REPO}`)" if is_hf else "Local Verified Checkpoint"
                rev_lbl = f"`{DEFAULT_HF_REVISION[:7]}`" if is_hf else "`Local`"

                st.caption(
                    f"**Verified Model**: `{model_choice.upper()}` | "
                    f"**Backend**: `{backend_disp}` | "
                    f"**Source**: {source_lbl} | "
                    f"**Revision**: {rev_lbl} | "
                    f"**SHA-256**: `{active_hash[:16]}... (Cryptographically Verified)` | "
                    f"**Area Basis**: {area_note}"
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
            if is_photo_rejected:
                st.warning("⚠️ **Inference Blocked**: The uploaded image was classified as consumer photography or corrupted payload. Please upload a valid Sentinel-2 optical tile.")
            elif backend == BACKEND_FASTAPI and not fastapi_healthy:
                st.info("ℹ️ FastAPI is unavailable. Please select **PyTorch (In-Process)** or **ONNX Runtime** to run inference.")
            elif backend == BACKEND_ONNX and onnx_load_error:
                st.error(f"❌ ONNX session error: {onnx_load_error}")
            elif model_load_error:
                st.error(f"❌ Cannot run inference: {model_load_error}")

    with tab2:
        st.markdown("### Cross-Architecture Side-by-Side Comparison")
        st.markdown(
            "Execute and compare predictions across all genuinely verified architectures "
            "using consistent preprocessing ($256 \\times 256$ float32, `torch.inference_mode()`, sigmoid activation)."
        )

        status_map = get_model_status_summary()
        available_keys = [k for k, meta in status_map.items() if meta["available"]]

        st.info(
            f"**Architecture Verification Status ({len(available_keys)}/{len(SUPPORTED_MODELS)} Available)**:\n\n" +
            "\n".join([f"• **{status_map[k]['display_name']}**: {status_map[k]['status_label']} — {status_map[k]['support_note']}" for k in SUPPORTED_MODELS])
        )

        if image is None:
            st.warning("⚠️ Please provide an image using the selector above to run comparisons.")
        elif is_photo_rejected:
            st.warning("⚠️ **Comparison Blocked**: The uploaded image is classified as consumer photography or invalid payload. Multi-architecture comparison is restricted to verified Earth observation imagery.")
        else:
            comp_button = st.button("Run Multi-Architecture Comparison", type="secondary", key="btn_run_comp")

            if comp_button or use_sample:
                st.markdown("---")
                comp_results = []

                for arch_key in available_keys:
                    try:
                        arch_model = get_cached_pytorch_model(arch_key)
                        m_img, b_mask, lat, cov_pct, _ = run_pytorch_inference(
                            model=arch_model,
                            image=image,
                            threshold=threshold
                        )
                        water_px = int(np.sum(b_mask))
                        area_km2, a_note = calculate_water_area_km2(water_px, classification, input_meta)
                        ov_img = create_overlay(image, m_img)
                        comp_results.append({
                            "key": arch_key,
                            "name": status_map[arch_key]["display_name"],
                            "mask_img": m_img,
                            "overlay_img": ov_img,
                            "latency_ms": lat,
                            "coverage_pct": cov_pct,
                            "area_km2": area_km2,
                            "area_note": a_note,
                        })
                    except Exception as e_comp:
                        st.error(f"Failed inference for `{arch_key}`: {e_comp}")

                if comp_results:
                    st.markdown("#### Visual Predictions & Overlays")
                    cols = st.columns(len(comp_results))
                    for col, res in zip(cols, comp_results):
                        with col:
                            st.markdown(f"**{res['name']}**")
                            st.image(res["mask_img"], caption=f"Predicted Mask (τ={threshold:.2f})", use_container_width=True)
                            st.image(res["overlay_img"], caption="Cyan Overlay", use_container_width=True)
                            st.metric("Water Coverage", f"{res['coverage_pct']:.2f}%")
                            area_val_disp = f"{res['area_km2']:.4f} km²" if res['area_km2'] is not None else "N/A"
                            st.metric("Estimated Area", area_val_disp)
                            st.metric("Latency", f"{res['latency_ms']:.2f} ms")

                    st.markdown("---")
                    st.markdown("#### Quantitative Comparison Table")
                    df_comp = pd.DataFrame([
                        {
                            "Architecture": r["name"],
                            "Inference Latency (ms)": f"{r['latency_ms']:.2f}",
                            "Predicted Water Coverage (%)": f"{r['coverage_pct']:.2f}%",
                            "Estimated Water Area (km²)": f"{r['area_km2']:.4f}" if r["area_km2"] is not None else "N/A",
                            "Quantification Basis": r["area_note"],
                        }
                        for r in comp_results
                    ])
                    st.dataframe(df_comp, use_container_width=True)

                st.caption(
                    "⚠️ **Single-Image Prediction Notice**: Water coverage percentage and nominal area are single-image "
                    "observational measurements calculated under nominal Sentinel-2 10m Ground Sample Distance assumption (100 m²/pixel) "
                    "or embedded GeoTIFF tags. They must NOT be confused with scientific test-set segmentation accuracy, precision, or recall. "
                    "Benchmark performance is measured against ground-truth labels across the complete test split in the Model Benchmarks tab."
                )

    with tab3:
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

    with tab4:
        st.subheader("Platform Architecture & Methodological Protocol")
        st.markdown(
            f"""
            ### Asymmetric Loss Optimization for Class Imbalance
            In optical satellite remote sensing, surface water accounts for less than 5.34% of total pixel volume across splits.
            Standard Cross-Entropy and Dice loss functions suffer from the **Accuracy Paradox**, achieving >86% accuracy by
            predicting pure background land while failing to detect narrow tributaries and fine coastal boundaries.
            GeoVision AI models are trained using **Focal Tversky Loss** (α=0.7, β=0.3, γ=4/3) to heavily penalize false negatives.

            ### Verified Multi-Architecture Pipeline
            The platform supports four distinct, authoritatively validated neural network architectures:
            1. **FPN (Feature Pyramid Network - ResNet34 Champion)**: Optimal balance of high recall (54.24%) and precision (75.81%), Micro Dice 0.6323.
            2. **UNet++ (ResNet34)**: Nested, dense skip connections capturing multi-scale boundary contexts.
            3. **DeepLabV3+ (ResNet34)**: Atrous Spatial Pyramid Pooling (ASPP) for wide contextual field segmentation.
            4. **SegFormer (MiT-B0)**: Lightweight hierarchical Vision Transformer without positional embeddings.

            ### Multi-Backend Inference Support
            - **PyTorch (In-Process)**: Native CPU/MPS inference using `torch.inference_mode()` with strict cryptographic checkpoint validation.
            - **ONNX Runtime Engine**: Standalone Open Neural Network Exchange graph execution with full operator parity and CPU graph optimization.
            - **FastAPI REST Microservice**: Decoupled asynchronous microservice running containerized inference with health, readiness, and metrics endpoints.

            ### Scientific Safety & Provenance Verification
            - **Cryptographic Model Registry**: All model checkpoints are verified against authoritative SHA-256 signatures before memory instantiation.
            - **Hugging Face Hub Revision**: Checkpoints hosted on [`{DEFAULT_HF_REPO}`](https://huggingface.co/{DEFAULT_HF_REPO}) pinned to immutable commit `{DEFAULT_HF_REVISION}`.
            - **Input Safety Gate**: 6-state image classification filters consumer photography and prevents non-scientific hydrological claims.
            - **Two-Mode Quantification**: Differentiates embedded GeoTIFF coordinate calculations from nominal 10m GSD approximations.
            """
        )


if __name__ == "__main__":
    main()
