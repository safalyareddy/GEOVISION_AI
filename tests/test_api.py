"""
GeoVision AI Satellite Water Body Segmentation Platform
Unit & Integration Tests for FastAPI Microservice
"""

import io
import pytest
import numpy as np
from PIL import Image

try:
    from fastapi.testclient import TestClient
    from src.api.main import app
    HAS_TESTCLIENT = True
except ImportError:
    HAS_TESTCLIENT = False


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_root_endpoint():
    """Test API root metadata endpoint."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "GeoVision AI — Water Body Segmentation API"
    assert data["status"] == "healthy"


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_health_check_endpoint():
    """Test API health check endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_list_models_endpoint():
    """Test listing available model architectures."""
    client = TestClient(app)
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "unet_plusplus" in data["supported_models"]


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_predict_endpoint_valid_image():
    """Test prediction endpoint with valid remote sensing image."""
    client = TestClient(app)

    # Create dummy synthetic RGB image with natural remote sensing reflectance (low saturation)
    img = Image.new("RGB", (64, 64), color=(60, 75, 70))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.png", buf, "image/png")},
        data={"model_name": "unet_plusplus", "threshold": 0.5}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "unet_plusplus"
    assert "water_coverage_percentage" in data
    assert "estimated_water_area_km2" in data
    assert "area_estimation_basis" in data
    assert "mask_base64" in data
    assert data["image_dimensions"] == [64, 64]


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_predict_endpoint_invalid_model():
    """Test prediction endpoint with unsupported model architecture."""
    client = TestClient(app)
    img = Image.new("RGB", (32, 32), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.png", buf, "image/png")},
        data={"model_name": "unsupported_arch", "threshold": 0.5}
    )
    assert response.status_code == 400
    assert "Invalid model" in response.json()["detail"]


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_predict_endpoint_invalid_threshold():
    """Test prediction endpoint with out-of-range threshold values."""
    client = TestClient(app)
    img = Image.new("RGB", (32, 32), color="green")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # Test threshold > 1.0
    res_high = client.post(
        "/predict",
        files={"file": ("test.png", buf, "image/png")},
        data={"model_name": "unet_plusplus", "threshold": 1.5}
    )
    assert res_high.status_code == 400

    # Test threshold < 0.0
    buf.seek(0)
    res_low = client.post(
        "/predict",
        files={"file": ("test.png", buf, "image/png")},
        data={"model_name": "unet_plusplus", "threshold": -0.2}
    )
    assert res_low.status_code == 400


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_predict_endpoint_empty_file():
    """Test prediction endpoint with empty 0-byte file."""
    client = TestClient(app)
    empty_buf = io.BytesIO(b"")

    response = client.post(
        "/predict",
        files={"file": ("empty.png", empty_buf, "image/png")},
        data={"model_name": "unet_plusplus", "threshold": 0.5}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_predict_endpoint_corrupt_file():
    """Test prediction endpoint with corrupt/malformed image bytes."""
    client = TestClient(app)
    corrupt_buf = io.BytesIO(b"NotARealPNGFileHeader1234567890")

    response = client.post(
        "/predict",
        files={"file": ("corrupt.png", corrupt_buf, "image/png")},
        data={"model_name": "unet_plusplus", "threshold": 0.5}
    )
    assert response.status_code == 400
    assert "corrupt" in response.json()["detail"].lower() or "decode" in response.json()["detail"].lower()


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_predict_endpoint_unsupported_mime():
    """Test prediction endpoint rejecting unsupported MIME type."""
    client = TestClient(app)
    buf = io.BytesIO(b"dummy text content")

    response = client.post(
        "/predict",
        files={"file": ("doc.txt", buf, "text/plain")},
        data={"model_name": "unet_plusplus", "threshold": 0.5}
    )
    assert response.status_code == 400
    assert "Unsupported image file format" in response.json()["detail"]


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_predict_endpoint_oversized_file():
    """Test prediction endpoint rejecting payload exceeding 10 MB limit."""
    client = TestClient(app)
    # 10 MB + 1 KB payload
    oversized_buf = io.BytesIO(b"X" * (10 * 1024 * 1024 + 1024))

    response = client.post(
        "/predict",
        files={"file": ("large.png", oversized_buf, "image/png")},
        data={"model_name": "unet_plusplus", "threshold": 0.5}
    )
    assert response.status_code == 413
    assert "exceeds maximum size limit" in response.json()["detail"].lower()


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_predict_endpoint_champion_fpn():
    """Test prediction endpoint with champion FPN model."""
    client = TestClient(app)
    img = Image.new("RGB", (64, 64), color=(60, 75, 70))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test_s2.png", buf, "image/png")},
        data={"model_name": "fpn", "threshold": 0.5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "fpn"
    assert "water_coverage_percentage" in data
    assert "latency_ms" in data
    assert "mask_base64" in data


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_models_endpoint_reports_champion_readiness():
    """Test that /models returns champion status and readiness details."""
    client = TestClient(app)
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["champion_model"] == "fpn"
    assert "models_status" in data
    assert "fpn" in data["models_status"]
    assert data["models_status"]["fpn"]["is_champion"] is True


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_predict_endpoint_all_four_models():
    """Test prediction endpoint returns valid results for all four supported models."""
    client = TestClient(app)
    img = Image.new("RGB", (128, 128), color=(65, 80, 75))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    for m_name in ["fpn", "unet_plusplus", "deeplabv3_plus", "segformer"]:
        response = client.post(
            "/predict",
            files={"file": (f"test_{m_name}.png", io.BytesIO(img_bytes), "image/png")},
            data={"model_name": m_name, "threshold": 0.5}
        )
        assert response.status_code == 200, f"Failed prediction for model '{m_name}': {response.text}"
        data = response.json()
        assert data["model_name"] == m_name
        assert 0.0 <= data["water_coverage_percentage"] <= 100.0
        assert data["latency_ms"] > 0
        assert len(data["mask_base64"]) > 0


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_predict_endpoint_rejects_consumer_photograph():
    """Test that consumer photographs (e.g., highly saturated floral/portrait images) are rejected with HTTP 422."""
    client = TestClient(app)
    # Pure red image (saturation 1.0) simulating consumer floral photography
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[..., 0] = 255
    arr[..., 1] = 10
    arr[..., 2] = 20
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("flower.png", buf, "image/png")},
        data={"model_name": "fpn", "threshold": 0.5}
    )
    assert response.status_code == 422
    assert "Unsupported photograph" in response.text


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient required for API tests.")
def test_ready_and_metrics_endpoints():
    """Test /ready, /metrics, and /models/{model_id}/status endpoints."""
    client = TestClient(app)
    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    data_ready = r_ready.json()
    assert "ready" in data_ready
    assert "service_state" in data_ready

    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200
    data_metrics = r_metrics.json()
    assert "metrics" in data_metrics
    assert "total_requests" in data_metrics["metrics"]

    r_status = client.get("/models/fpn/status")
    assert r_status.status_code == 200
    assert r_status.json()["model_name"] == "fpn"

    r_status_bad = client.get("/models/nonexistent_model/status")
    assert r_status_bad.status_code == 404