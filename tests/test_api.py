"""
GeoVision AI Satellite Water Body Segmentation Platform
Unit & Integration Tests for FastAPI Microservice
"""

import io
import pytest
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
    """Test prediction endpoint with synthetic PNG image."""
    client = TestClient(app)

    # Create dummy synthetic RGB image
    img = Image.new("RGB", (64, 64), color="blue")
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