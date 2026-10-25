import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

def test_api_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["ml_models_loaded"] == 0

def test_api_capabilities():
    resp = client.get("/api/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["forensic_modules"]) >= 7

def test_api_upload_and_analyze():
    jpg_path = FIXTURES_DIR / "pristine_sample.jpg"
    with open(jpg_path, "rb") as f:
        resp = client.post("/api/analyze", files={"file": ("pristine.jpg", f, "image/jpeg")})
    assert resp.status_code == 200
    data = resp.json()
    assert "analysis_id" in data
    assert "verdict" in data
    assert "module_results" in data
    assert "ela" in data["module_results"]
    assert "fft" in data["module_results"]

    analysis_id = data["analysis_id"]
    # Check retrieval endpoint
    resp_get = client.get(f"/api/analysis/{analysis_id}")
    assert resp_get.status_code == 200
    assert resp_get.json()["analysis_id"] == analysis_id

def test_api_dataset_status():
    resp = client.get("/api/dataset/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
