# URK24CS1111
# QuantumSense - Tests for Flask Web Dashboard & REST API

"""
test_dashboard.py - Pytest test suite for Flask web dashboard backend.
Validates HTTP endpoints, JSON REST responses, model routing, feature generation,
error responses for missing metrics and invalid model types, and response schema.
"""

import os
import sys
import json
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config as cfg
from src.config import FEATURE_COLUMNS, FAULT_CLASSES
from dashboard.app import app
import dashboard.app as dashboard_module


@pytest.fixture
def client():
    """Create Flask test client in test mode."""
    app.config["TESTING"] = True
    app.config["DEBUG"] = False
    with app.test_client() as test_client:
        yield test_client


def test_home_page_loads(client):
    """Verify GET / returns HTTP 200 status code."""
    response = client.get("/")
    assert response.status_code == 200


def test_home_page_has_title(client):
    """Verify response from GET / contains the application title 'QuantumSense'."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"QuantumSense" in response.data


def test_api_status(client):
    """Verify GET /api/status returns JSON payload with expected system status keys."""
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.is_json

    data = response.get_json()
    assert data["success"] is True
    assert "models" in data
    assert "classical_svm" in data["models"]
    assert "quantum_qsvc" in data["models"]
    assert "fault_classes" in data
    assert "total_samples" in data
    assert "timestamp" in data


def test_api_metrics_no_file(client, monkeypatch):
    """Verify GET /api/metrics returns 404 error when metrics.json does not exist."""
    missing_metrics_path = "non_existent_metrics_file.json"
    monkeypatch.setattr(cfg, "METRICS_JSON", missing_metrics_path)
    monkeypatch.setattr(dashboard_module, "METRICS_JSON", missing_metrics_path)

    response = client.get("/api/metrics")
    assert response.status_code == 404
    assert response.is_json

    data = response.get_json()
    assert data["success"] is False
    assert data["available"] is False
    assert "error" in data


def test_api_generate(client):
    """Verify POST /api/generate produces a simulated sensor window with 15 features."""
    response = client.post("/api/generate", json={"fault_class": 1})
    assert response.status_code == 200
    assert response.is_json

    data = response.get_json()
    assert data["success"] is True
    assert "sample" in data

    sample = data["sample"]
    assert "features" in sample
    assert "temperature" in sample
    assert "pressure" in sample
    assert "vibration" in sample

    features = sample["features"]
    for col in FEATURE_COLUMNS:
        assert col in features, f"Generated sample missing feature column '{col}'."


def test_api_predict_classical(client, monkeypatch, sample_feature_dict):
    """Verify POST /api/predict successfully routes classical model requests."""
    # Mock prediction call to avoid disk model dependency
    def mock_predict_classical(features):
        return {
            "class": 0,
            "label": "Normal",
            "confidence": 0.9654,
        }

    monkeypatch.setattr(dashboard_module, "call_predict_classical", mock_predict_classical)

    payload = {
        "features": sample_feature_dict,
        "model_type": "classical",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    assert response.is_json

    data = response.get_json()
    assert data["success"] is True
    assert data["model_type"] == "classical"
    assert data["prediction"]["class"] == 0
    assert data["prediction"]["label"] == "Normal"


def test_api_predict_invalid_model(client, sample_feature_dict):
    """Verify POST /api/predict rejects invalid model_type with HTTP 400."""
    payload = {
        "features": sample_feature_dict,
        "model_type": "deep_neural_network",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert response.is_json

    data = response.get_json()
    assert data["success"] is False
    assert "error" in data
    assert "Invalid model_type" in data["error"]


def test_api_predict_missing_features(client, monkeypatch):
    """Verify POST /api/predict rejects empty or missing features with HTTP 400."""
    # Ensure history buffer is cleared so it cannot fallback
    monkeypatch.setattr(dashboard_module, "recent_sensor_readings", [])

    payload = {
        "features": {},
        "model_type": "classical",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert response.is_json

    data = response.get_json()
    assert data["success"] is False
    assert "error" in data


def test_api_sensor_history(client):
    """Verify GET /api/sensor_history returns a list of recent sensor readings."""
    # Seed history buffer by generating a sample first
    client.post("/api/generate", json={"fault_class": 0})

    response = client.get("/api/sensor_history?limit=10")
    assert response.status_code == 200
    assert response.is_json

    data = response.get_json()
    assert data["success"] is True
    assert "history" in data
    assert isinstance(data["history"], list)
    assert len(data["history"]) >= 1


def test_api_evaluation(client, monkeypatch, tmp_path):
    """Verify GET /api/evaluation returns metrics and evaluation metadata."""
    dummy_eval = {
        "classical": {
            "accuracy": 0.94,
            "precision": 0.93,
            "recall": 0.92,
            "f1": 0.925,
        },
        "quantum": {
            "accuracy": 0.88,
            "precision": 0.87,
            "recall": 0.86,
            "f1": 0.865,
        },
    }
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps(dummy_eval), encoding="utf-8")

    monkeypatch.setattr(cfg, "METRICS_JSON", str(metrics_file))
    monkeypatch.setattr(dashboard_module, "METRICS_JSON", str(metrics_file))

    response = client.get("/api/evaluation")
    assert response.status_code == 200
    assert response.is_json

    data = response.get_json()
    assert data["success"] is True
    assert "evaluation" in data
    assert "classical" in data["evaluation"]
    assert "quantum" in data["evaluation"]
    assert "images" in data["evaluation"]


def test_prediction_response_format(client, monkeypatch, sample_feature_dict):
    """Verify prediction response schema includes success, prediction, model_used, timestamp."""
    def mock_predict(features):
        return {
            "class": 2,
            "label": "Stuck",
            "confidence": 0.91,
        }

    monkeypatch.setattr(dashboard_module, "call_predict_classical", mock_predict)

    payload = {
        "features": sample_feature_dict,
        "model_type": "classical",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    assert "success" in data
    assert data["success"] is True
    assert "prediction" in data
    assert "class" in data["prediction"]
    assert "label" in data["prediction"]
    assert "confidence" in data["prediction"]
    assert "model_used" in data
    assert "timestamp" in data
    assert data["prediction"]["class"] == 2
    assert data["prediction"]["label"] == "Stuck"
