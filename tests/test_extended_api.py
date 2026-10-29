# URK24CS1111
"""
test_extended_api.py - Pytest suite for extended Flask API endpoints:
- /api/dataset/upload
- /api/dataset/process
- /api/dataset/active
- /api/dataset/select_source
- /api/dataset/analytics
- /api/train, /api/train/status, /api/train/cancel
- /api/predict_batch
- /api/download/<download_type>
- /api/experiments and /api/experiments/clear
"""

import io
import json
import pytest
import pandas as pd
from src.config import FEATURE_COLUMNS


def test_api_dataset_upload_valid_csv(client):
    """Test /api/dataset/upload with a valid multipart CSV file."""
    csv_data = "temperature,pressure,vibration,fault_label\n75.1,101.2,0.50,Normal\n78.2,100.9,0.63,Drift\n"
    data = {
        "file": (io.BytesIO(csv_data.encode("utf-8")), "test_telemetry.csv")
    }
    response = client.post("/api/dataset/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_json = response.get_json()
    assert res_json["success"] is True
    assert res_json["total_rows"] == 2
    col_names = [c["name"] for c in res_json["columns"]]
    assert "temperature" in col_names


def test_api_dataset_upload_invalid_extension(client):
    """Test /api/dataset/upload rejects non-csv files."""
    txt_data = "some text file content"
    data = {
        "file": (io.BytesIO(txt_data.encode("utf-8")), "test.txt")
    }
    response = client.post("/api/dataset/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res_json = response.get_json()
    assert res_json["success"] is False
    assert "CSV" in res_json["error"]


def test_api_dataset_process_and_active(client):
    """Test /api/dataset/process processes an uploaded file and registers it as active."""
    # First upload
    csv_data = (
        "temp_mean,pres_mean,vib_mean,fault_class\n"
        "75.1,101.2,0.50,0\n"
        "74.8,101.5,0.48,0\n"
        "78.2,100.9,0.63,1\n"
        "79.1,100.5,0.72,1\n"
    )
    up_data = {
        "file": (io.BytesIO(csv_data.encode("utf-8")), "proc_test.csv")
    }
    up_res = client.post("/api/dataset/upload", data=up_data, content_type="multipart/form-data")
    assert up_res.status_code == 200
    filename = up_res.get_json()["uploaded_filename"]

    # Now process
    proc_payload = {
        "filename": filename,
        "mode": "precomputed",
        "feature_columns": ["temp_mean", "pres_mean", "vib_mean"],
        "target_column": "fault_class",
        "missing_strategy": "mean",
        "drop_duplicates": True,
        "handle_outliers": False
    }
    proc_res = client.post("/api/dataset/process", json=proc_payload)
    assert proc_res.status_code == 200
    proc_json = proc_res.get_json()
    assert proc_json["success"] is True
    assert proc_json["total_samples"] == 4
    assert proc_json["feature_count"] == 3

    # Check active dataset info
    act_res = client.get("/api/dataset/active")
    assert act_res.status_code == 200
    act_json = act_res.get_json()
    assert act_json["source"] == "custom"
    assert act_json["is_custom"] is True

    # Switch back to synthetic
    sel_res = client.post("/api/dataset/select_source", json={"source": "synthetic"})
    assert sel_res.status_code == 200
    sel_json = sel_res.get_json()
    assert sel_json["success"] is True
    assert sel_json["active_dataset"]["source"] == "synthetic"


def test_api_dataset_analytics(client):
    """Test /api/dataset/analytics returns valid statistical payload."""
    res = client.get("/api/dataset/analytics")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "total_rows" in data
    assert "feature_statistics" in data
    assert "correlation" in data


def test_api_train_status_and_cancel(client):
    """Test /api/train/status returns dictionary with progress and logs."""
    status_res = client.get("/api/train/status")
    assert status_res.status_code == 200
    status_data = status_res.get_json()
    assert "status" in status_data
    assert "progress" in status_data
    assert "stage" in status_data

    # Test cancel endpoint when idle
    cancel_res = client.post("/api/train/cancel")
    assert cancel_res.status_code == 200


def test_api_predict_batch(client):
    """Test /api/predict_batch accepts a CSV and returns predictions preview."""
    from models.classical_model import train_classical_model
    train_classical_model()

    # Generate sample row with all 15 feature columns
    header = ",".join(FEATURE_COLUMNS)
    row_1 = ",".join(["75.0" if "temp" in c else ("101.3" if "pres" in c else "0.50") for c in FEATURE_COLUMNS])
    row_2 = ",".join(["80.0" if "temp" in c else ("99.0" if "pres" in c else "0.85") for c in FEATURE_COLUMNS])
    csv_data = f"{header}\n{row_1}\n{row_2}\n"

    data = {
        "file": (io.BytesIO(csv_data.encode("utf-8")), "batch_input.csv"),
        "model_type": "classical"
    }
    res = client.post("/api/predict_batch", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    res_json = res.get_json()
    assert res_json["success"] is True
    assert res_json["total_rows"] == 2
    assert len(res_json["preview"]) == 2
    assert "predicted_label" in res_json["preview"][0]


def test_api_downloads(client):
    """Test all download endpoints return appropriate content or attachments."""
    # Cleaned/synthetic dataset
    dl_data = client.get("/api/download/cleaned_dataset")
    assert dl_data.status_code in [200, 404]

    # Metrics CSV
    dl_metrics = client.get("/api/download/metrics_csv")
    assert dl_metrics.status_code in [200, 404]

    # Experiment history CSV
    dl_exp = client.get("/api/download/experiment_history")
    assert dl_exp.status_code == 200
    assert "text/csv" in dl_exp.content_type


def test_api_experiments_list_and_clear(client):
    """Test /api/experiments returns array and /api/experiments/clear empties it."""
    res = client.get("/api/experiments")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert isinstance(data["experiments"], list)

    clear_res = client.post("/api/experiments/clear")
    assert clear_res.status_code == 200
    clear_data = clear_res.get_json()
    assert clear_data["success"] is True
