# Phase Seven — Streamlit Dashboard & Frontend Audit

**Audit Date**: October 12, 2026  
**Auditor**: Staff Frontend & QA Automation Engineer  
**Dashboard Path**: [`app.py`](file:///Users/safalyareddy/Desktop/Projects/GeoVision-AI-Satellite-Water-Body-Segmentation-Platform/app.py)  
**Live Service URL**: `http://localhost:8501`  

---

## 1. UI Architecture & Functional Capabilities

The Streamlit dashboard provides an intuitive, responsive computer vision workstation for earth observation analysts:

1. **Architecture & Runtime Selection**:
   * Dropdown selector for all three supported models (`UNet++`, `DeepLabV3+`, `SegFormer`).
   * Toggle between PyTorch CPU/MPS and ONNX Runtime CPU execution.
2. **Interactive Threshold Slider**:
   * Dynamic thresholding from $0.05$ to $0.95$ (default: $0.50$), enabling interactive sensitivity tuning for turbid water or shallow wetlands.
3. **Multi-View Visual Inspection**:
   * Columns displaying:
     1. Raw Input Satellite RGB Image
     2. Predicted Binary Water Mask (black & white)
     3. High-Contrast Overlay (semi-transparent cyan mask projected over the input image)
4. **Hydrological Area Estimation**:
   * Displays water coverage percentage.
   * Reports estimated surface water area in both square kilometers ($km^2$) and hectares, explicitly qualified with a label: *"Calibrated nominal estimate assuming Sentinel-2 10m GSD"*.

---

## 2. End-to-End Verification & Browser Validation

* **Browser Session State**: Verified live in browser with WebP demo recording (`geovision_web_demo.webp`).
* **Error Resilience**:
  * Clean empty-state prompt when awaiting user tile upload.
  * Informative diagnostics if an ONNX model or PyTorch checkpoint is absent.
  * Responsive layout tested on desktop ($1470 \times 776$) and narrow mobile viewports.
