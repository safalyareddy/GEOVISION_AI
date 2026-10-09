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
