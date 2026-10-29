# URK24CS1111
"""
Unit tests for Custom Dataset Lab and Experiment Tracker modules:
- CSV inspection and column type detection
- Sliding-window time-series feature engineering (Mode B)
- Data cleaning, imputation, outlier handling, and label encoding (Mode A & B)
- Dataset analytics, statistical properties, and correlation matrix
- Active dataset state switching
- Experiment history logging and auditing
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.custom_dataset import (
    inspect_csv,
    extract_window_features,
    clean_and_preprocess,
    compute_dataset_analytics,
    get_active_dataset_info,
    set_active_dataset_source,
    save_custom_dataset,
    get_active_dataframe,
)
from src.experiment_tracker import (
    log_experiment,
    get_experiments,
    clear_experiments,
    export_history_dataframe,
)


@pytest.fixture
def sample_timeseries_df():
    """Create a sample sequential time-series DataFrame with 100 timesteps."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1s"),
        "temperature": 75.0 + np.random.normal(0, 1.0, n),
        "pressure": 101.3 + np.random.normal(0, 0.5, n),
        "vibration": 0.5 + np.random.normal(0, 0.05, n),
        "fault_condition": ["Normal"] * 50 + ["Drift"] * 50
    })
    return df


@pytest.fixture
def sample_precomputed_df():
    """Create a sample precomputed feature DataFrame with missing values and duplicates."""
    np.random.seed(42)
    df = pd.DataFrame({
        "temp_mean": [74.5, 75.2, 75.0, 78.1, np.nan, 75.0],
        "pres_mean": [101.2, 101.5, 100.9, 103.0, 101.1, 100.9],
        "vib_mean": [0.49, 0.51, 0.48, 0.95, 0.50, 0.48],
        "fault_label": ["Normal", "Normal", "Normal", "Drift", "Normal", "Normal"]
    })
    return df


def test_inspect_csv_valid(tmp_path, sample_precomputed_df):
    """Test inspect_csv parses rows, columns, and types accurately."""
    csv_path = tmp_path / "test_inspect.csv"
    sample_precomputed_df.to_csv(csv_path, index=False)

    report = inspect_csv(str(csv_path))
    assert report["success"] is True
    assert report["total_rows"] == 6
    assert report["total_columns"] == 4
    col_names = [c["name"] for c in report["columns"]]
    assert "temp_mean" in col_names
    assert "fault_label" in col_names
    assert "temp_mean" in report["numeric_columns"]
    assert report["missing_values_total"] == 1
    assert report["has_labels"] is True
    assert report["suggested_target"] == "fault_label"


def test_inspect_csv_missing_file():
    """Test inspect_csv handles non-existent file cleanly."""
    with pytest.raises(FileNotFoundError):
        inspect_csv("non_existent_file_path_123.csv")


def test_extract_window_features(sample_timeseries_df):
    """Test Mode B sliding-window feature engineering generates 5 features per sensor."""
    window_size = 20
    extracted = extract_window_features(
        df=sample_timeseries_df,
        sensor_columns=["temperature", "pressure", "vibration"],
        window_size=window_size,
        target_col="fault_condition",
        timestamp_col="timestamp"
    )

    # 100 samples with window_size 20 -> 5 non-overlapping windows
    assert len(extracted) == 5
    expected_cols = [
        "temperature_mean", "temperature_std", "temperature_min", "temperature_max", "temperature_slope",
        "pressure_mean", "pressure_std", "pressure_min", "pressure_max", "pressure_slope",
        "vibration_mean", "vibration_std", "vibration_min", "vibration_max", "vibration_slope",
        "fault_label"
    ]
    for col in expected_cols:
        assert col in extracted.columns, f"Missing expected extracted column: {col}"

    assert not extracted.isnull().values.any()


def test_clean_and_preprocess_impute_and_duplicates(sample_precomputed_df):
    """Test Mode A cleaning handles missing value imputation and removes duplicate rows."""
    clean_df, mapping, report = clean_and_preprocess(
        df=sample_precomputed_df,
        feature_columns=["temp_mean", "pres_mean", "vib_mean"],
        target_column="fault_label",
        missing_strategy="mean",
        drop_duplicates=True,
        handle_outliers=False
    )

    # Duplicates removed: row 5 was identical to row 2
    assert len(clean_df) < len(sample_precomputed_df)
    # Missing values in temp_mean imputed with mean
    assert clean_df["temp_mean"].isnull().sum() == 0
    # Categorical label encoded to integer fault_class
    assert "fault_class" in clean_df.columns
    assert isinstance(clean_df["fault_class"].iloc[0], (int, np.integer))
    assert 0 in mapping
    assert report["initial_rows"] == 6
    assert report["final_rows"] == len(clean_df)
    assert report["has_labels"] is True


def test_compute_dataset_analytics(sample_timeseries_df):
    """Test statistical metrics, class distribution, and correlation matrix calculation."""
    analytics = compute_dataset_analytics(
        df=sample_timeseries_df,
        feature_columns=["temperature", "pressure", "vibration"],
        target_column="fault_condition"
    )

    assert analytics["success"] is True
    assert analytics["total_rows"] == 100
    assert analytics["feature_count"] == 3
    assert "feature_statistics" in analytics
    stat_cols = [s["feature"] for s in analytics["feature_statistics"]]
    assert "temperature" in stat_cols
    assert "correlation" in analytics
    assert "class_distribution" in analytics
    assert analytics["class_distribution"]["Normal"] == 50
    assert analytics["class_distribution"]["Drift"] == 50


def test_active_dataset_source_toggle():
    """Test selecting dataset source between synthetic and custom."""
    info_synth = set_active_dataset_source("synthetic")
    assert info_synth["source"] == "synthetic"
    assert info_synth["is_custom"] is False

    active_now = get_active_dataset_info()
    assert active_now["source"] == "synthetic"

    meta = {
        "dataset_name": "Test Custom Unit",
        "source": "custom",
        "feature_columns": ["f1", "f2"],
        "target_column": "fault_class",
        "fault_classes": {0: "Normal", 1: "Fault"},
        "cleaning_report": {"has_labels": True}
    }
    dummy_df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "fault_class": [0, 1]})
    save_custom_dataset(dummy_df, meta)

    info_cust = set_active_dataset_source("custom")
    assert info_cust["source"] == "custom"
    assert info_cust["is_custom"] is True

    # Revert back to synthetic to leave clean state
    set_active_dataset_source("synthetic")
    assert get_active_dataset_info()["source"] == "synthetic"


def test_experiment_tracker():
    """Test logging an experiment, reading it back, and clearing history."""
    clear_experiments()
    initial = get_experiments()
    assert len(initial) == 0

    exp_data = {
        "dataset_name": "Test Suite Benchmark",
        "dataset_source": "synthetic",
        "model_type": "Classical SVM",
        "features_count": 15,
        "train_samples": 800,
        "test_samples": 200,
        "accuracy": 0.985,
        "precision": 0.980,
        "recall": 0.982,
        "macro_f1": 0.981,
        "weighted_f1": 0.981,
        "train_time_sec": 0.052,
        "predict_time_sec": 0.002,
        "status": "SUCCESS"
    }
    logged = log_experiment(exp_data)
    assert logged["experiment_id"].startswith("EXP-")
    assert logged["accuracy"] == 0.985

    all_exps = get_experiments()
    assert len(all_exps) == 1
    assert all_exps[0]["model_type"] == "Classical SVM"

    df_hist = export_history_dataframe()
    assert len(df_hist) == 1
    assert "experiment_id" in df_hist.columns

    cleared = clear_experiments()
    assert cleared is True
    assert len(get_experiments()) == 0
