# URK24CS1111
# QuantumSense - Pytest Configuration and Shared Test Fixtures

"""
conftest.py - Pytest fixtures and environment configuration for QuantumSense tests.
Provides shared dataset samples, mock models, Flask test clients, and test markers.
"""

import os
import sys
from typing import Dict, Generator
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config as cfg
from src.config import (
    FEATURE_COLUMNS,
    FAULT_CLASSES,
    NUM_CLASSES,
    TARGET_COLUMN,
    LABEL_COLUMN,
    TEMP_MEAN,
    TEMP_STD,
    PRESSURE_MEAN,
    PRESSURE_STD,
    VIBRATION_MEAN,
    VIBRATION_STD,
)


def pytest_configure(config):
    """Register custom markers for slow and quantum tests."""
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "quantum: marks tests that execute quantum circuit simulations")


@pytest.fixture(scope="session")
def qiskit_available() -> bool:
    """Check if Qiskit and Qiskit Machine Learning are installed and usable."""
    try:
        import qiskit
        import qiskit_machine_learning
        from qiskit.circuit.library import ZZFeatureMap
        return True
    except (ImportError, ModuleNotFoundError, Exception):
        return False


@pytest.fixture
def sample_feature_dict() -> Dict[str, float]:
    """Return a dictionary of realistic nominal sensor features for testing."""
    return {
        "temp_mean": float(TEMP_MEAN),
        "temp_std": float(TEMP_STD),
        "temp_min": float(TEMP_MEAN - 2.0),
        "temp_max": float(TEMP_MEAN + 2.0),
        "temp_slope": 0.005,
        "pres_mean": float(PRESSURE_MEAN),
        "pres_std": float(PRESSURE_STD),
        "pres_min": float(PRESSURE_MEAN - 1.5),
        "pres_max": float(PRESSURE_MEAN + 1.5),
        "pres_slope": -0.002,
        "vib_mean": float(VIBRATION_MEAN),
        "vib_std": float(VIBRATION_STD),
        "vib_min": float(max(0.0, VIBRATION_MEAN - 0.1)),
        "vib_max": float(VIBRATION_MEAN + 0.1),
        "vib_slope": 0.0001,
    }


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    Generate a small synthetic DataFrame (50 samples, 10 per class)
    with all required columns for fast testing without touching the disk.
    """
    rng = np.random.RandomState(42)
    rows = []
    samples_per_class = 10

    for fault_class in sorted(FAULT_CLASSES.keys()):
        for _ in range(samples_per_class):
            row = {}
            for col in FEATURE_COLUMNS:
                if "temp" in col:
                    base = TEMP_MEAN if "mean" in col or "min" in col or "max" in col else TEMP_STD
                elif "pres" in col:
                    base = PRESSURE_MEAN if "mean" in col or "min" in col or "max" in col else PRESSURE_STD
                else:
                    base = VIBRATION_MEAN if "mean" in col or "min" in col or "max" in col else VIBRATION_STD

                if "slope" in col:
                    row[col] = float(rng.normal(0.0, 0.05))
                else:
                    row[col] = float(rng.normal(base, 0.2))

            row[TARGET_COLUMN] = int(fault_class)
            row[LABEL_COLUMN] = FAULT_CLASSES[fault_class]
            rows.append(row)

    df = pd.DataFrame(rows)
    return df[FEATURE_COLUMNS + [TARGET_COLUMN, LABEL_COLUMN]]


@pytest.fixture
def temporary_dataset_csv(tmp_path, sample_dataframe) -> str:
    """Save the sample DataFrame into a temporary CSV file and return the path."""
    csv_path = tmp_path / "sensor_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def quantum_small_config(monkeypatch):
    """
    Monkeypatch configuration constants to very small numbers (10 train, 5 test)
    so quantum tests execute in a few seconds rather than minutes.
    """
    monkeypatch.setattr(cfg, "QUANTUM_TRAIN_SAMPLES", 10)
    monkeypatch.setattr(cfg, "QUANTUM_TEST_SAMPLES", 5)

    import src.preprocess as pp
    monkeypatch.setattr(pp, "QUANTUM_TRAIN_SAMPLES", 10)
    monkeypatch.setattr(pp, "QUANTUM_TEST_SAMPLES", 5)

    import models.quantum_model as qm
    monkeypatch.setattr(qm, "QUANTUM_TRAIN_SAMPLES", 10)
    monkeypatch.setattr(qm, "QUANTUM_TEST_SAMPLES", 5)


@pytest.fixture
def client() -> Generator:
    """
    Create a Flask test client for the dashboard application.
    Enables testing mode and disables exception catching for clean error traces.
    """
    from dashboard.app import app
    app.config["TESTING"] = True
    app.config["DEBUG"] = False
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture(scope="session", autouse=True)
def ensure_test_dataset():
    """Ensure baseline sensor_data.csv exists for model training tests."""
    from src.config import SENSOR_DATA_CSV
    if not os.path.exists(SENSOR_DATA_CSV):
        from src.generate_data import generate_dataset
        generate_dataset(samples_per_class=20, save_csv=True, save_plot=False)

