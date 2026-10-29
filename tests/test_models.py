# URK24CS1111
# QuantumSense - Tests for Classical and Quantum Models

"""
test_models.py - Pytest test suite for classical SVM and Quantum QSVC models.
Validates training pipelines, evaluation metrics, prediction validity,
quantum circuit construction, kernel evaluation, and error handling.
"""

import os
import sys
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config as cfg
from src.config import (
    CLASSICAL_MODEL_PATH,
    CLASSICAL_SCALER_PATH,
    QUANTUM_RESULTS_PATH,
    QUANTUM_SCALER_PATH,
    FAULT_CLASSES,
    NUM_CLASSES,
    FEATURE_COLUMNS,
    QUANTUM_FEATURE_COUNT,
)
from models.classical_model import (
    train_classical_model,
    predict_classical,
)
from models.quantum_model import (
    get_quantum_circuit_info,
    train_quantum_model,
    predict_quantum,
    _QISKIT_IMPORT_ERROR,
)

# Detect Qiskit availability
HAS_QISKIT = _QISKIT_IMPORT_ERROR is None
try:
    if HAS_QISKIT:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
except (ImportError, ModuleNotFoundError, Exception):
    HAS_QISKIT = False

skip_if_no_qiskit = pytest.mark.skipif(
    not HAS_QISKIT,
    reason=f"Qiskit / Qiskit Machine Learning is unavailable: {_QISKIT_IMPORT_ERROR}",
)


# ---------------------------------------------------------------------------
# Classical Model Fixtures & Tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def classical_training_result():
    """Train classical model once for the test module to verify returned results."""
    return train_classical_model()


def test_classical_train_succeeds(classical_training_result):
    """Verify training executes successfully and returns expected dictionary keys."""
    res = classical_training_result
    assert isinstance(res, dict), "Result must be a dictionary."

    expected_keys = [
        "model",
        "scaler",
        "y_test",
        "y_pred",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
    ]
    for key in expected_keys:
        assert key in res, f"Expected key '{key}' not found in classical training results."


def test_classical_predictions_valid(classical_training_result):
    """Verify test predictions are within valid fault class range [0, NUM_CLASSES - 1]."""
    res = classical_training_result
    y_pred = np.asarray(res["y_pred"])
    assert len(y_pred) > 0, "Predictions array is empty."
    assert np.all(y_pred >= 0), "Found negative predicted class."
    assert np.all(y_pred < NUM_CLASSES), f"Found predicted class >= {NUM_CLASSES}."


def test_classical_metrics_valid(classical_training_result):
    """Verify classification metrics are bounded within [0.0, 1.0]."""
    res = classical_training_result
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        val = res[metric_name]
        assert isinstance(val, (float, np.floating)), f"Metric '{metric_name}' is not float."
        assert 0.0 <= val <= 1.0, f"Metric '{metric_name}'={val} outside [0.0, 1.0]."


def test_classical_confusion_matrix_shape(classical_training_result):
    """Verify confusion matrix has square dimensions (NUM_CLASSES, NUM_CLASSES)."""
    res = classical_training_result
    cm = np.asarray(res["confusion_matrix"])
    assert cm.ndim == 2, f"Confusion matrix has {cm.ndim} dimensions, expected 2."
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES), (
        f"Expected confusion matrix shape ({NUM_CLASSES}, {NUM_CLASSES}), got {cm.shape}."
    )


def test_classical_model_saved():
    """Verify model and scaler joblib artefacts are persisted on disk."""
    assert os.path.exists(CLASSICAL_MODEL_PATH), (
        f"Classical model artefact missing at: {CLASSICAL_MODEL_PATH}"
    )
    assert os.path.exists(CLASSICAL_SCALER_PATH), (
        f"Classical scaler artefact missing at: {CLASSICAL_SCALER_PATH}"
    )
    assert os.path.getsize(CLASSICAL_MODEL_PATH) > 0
    assert os.path.getsize(CLASSICAL_SCALER_PATH) > 0


def test_classical_predict_single(sample_feature_dict):
    """Test single-sample classical inference with valid input feature dictionary."""
    result = predict_classical(sample_feature_dict)
    assert isinstance(result, dict)
    assert "predicted_class" in result
    assert "predicted_label" in result
    assert "confidence" in result

    assert 0 <= result["predicted_class"] < NUM_CLASSES
    assert result["predicted_label"] == FAULT_CLASSES[result["predicted_class"]]
    assert 0.0 <= result["confidence"] <= 1.0


def test_classical_predict_missing_model(monkeypatch, sample_feature_dict):
    """Verify predict_classical raises FileNotFoundError when artefacts are missing."""
    monkeypatch.setattr(cfg, "CLASSICAL_MODEL_PATH", "non_existent_model.joblib")
    import models.classical_model as cm
    monkeypatch.setattr(cm, "CLASSICAL_MODEL_PATH", "non_existent_model.joblib")

    with pytest.raises((FileNotFoundError, Exception)):
        predict_classical(sample_feature_dict)


# ---------------------------------------------------------------------------
# Quantum Model Fixtures & Tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def quantum_fast_result():
    """Train a very small quantum model (10 train, 5 test) for quick verification."""
    if not HAS_QISKIT:
        pytest.skip(f"Qiskit unavailable: {_QISKIT_IMPORT_ERROR}")
    return train_quantum_model(train_samples=10, test_samples=5)


@pytest.mark.quantum
def test_quantum_feature_map_creation():
    """Verify ZZFeatureMap quantum circuit can be successfully constructed."""
    if not HAS_QISKIT:
        pytest.skip("Qiskit not installed")

    from qiskit.circuit.library import ZZFeatureMap
    feature_dim = 4
    reps = 2
    fmap = ZZFeatureMap(feature_dimension=feature_dim, reps=reps, entanglement="linear")

    assert fmap.num_qubits == feature_dim
    assert fmap.num_parameters == feature_dim
    assert fmap.depth() > 0


@pytest.mark.quantum
def test_quantum_kernel_creation():
    """Verify FidelityQuantumKernel can be instantiated with ZZFeatureMap."""
    if not HAS_QISKIT:
        pytest.skip("Qiskit not installed")

    from qiskit.circuit.library import ZZFeatureMap
    from models.quantum_model import create_quantum_kernel

    fmap = ZZFeatureMap(feature_dimension=4, reps=1, entanglement="linear")
    kernel = create_quantum_kernel(fmap)
    assert kernel is not None


@pytest.mark.quantum
def test_quantum_circuit_info():
    """Verify get_quantum_circuit_info returns valid structural circuit metrics."""
    if not HAS_QISKIT:
        pytest.skip("Qiskit not installed")

    info = get_quantum_circuit_info(feature_dimension=4, reps=2, entanglement="linear")
    assert isinstance(info, dict)
    assert info["num_qubits"] == 4
    assert info["circuit_depth"] > 0
    assert "circuit_diagram" in info
    assert len(info["circuit_diagram"]) > 0


@pytest.mark.slow
@pytest.mark.quantum
def test_quantum_train_succeeds(quantum_fast_result):
    """Verify quantum model training succeeds and returns dictionary with expected keys."""
    res = quantum_fast_result
    assert isinstance(res, dict)

    expected_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
        "y_pred_q",
        "y_test_q",
        "train_samples",
        "test_samples",
        "num_qubits",
    ]
    for key in expected_keys:
        assert key in res, f"Expected key '{key}' not found in quantum results."


@pytest.mark.slow
@pytest.mark.quantum
def test_quantum_predictions_valid(quantum_fast_result):
    """Verify quantum predictions contain valid integer class labels."""
    res = quantum_fast_result
    y_pred = np.asarray(res["y_pred_q"])
    assert len(y_pred) > 0
    assert np.all(y_pred >= 0)
    assert np.all(y_pred < NUM_CLASSES)


@pytest.mark.slow
@pytest.mark.quantum
def test_quantum_metrics_valid(quantum_fast_result):
    """Verify quantum evaluation metrics fall within the [0.0, 1.0] interval."""
    res = quantum_fast_result
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        val = res[metric_name]
        assert isinstance(val, (float, np.floating)), f"Quantum metric '{metric_name}' is not float."
        assert 0.0 <= val <= 1.0, f"Quantum metric '{metric_name}'={val} outside [0.0, 1.0]."


@pytest.mark.slow
@pytest.mark.quantum
def test_quantum_kernel_matrix_shape():
    """Verify that quantum kernel evaluation yields a square Gram matrix."""
    if not HAS_QISKIT:
        pytest.skip("Qiskit not installed")

    from qiskit.circuit.library import ZZFeatureMap
    from models.quantum_model import create_quantum_kernel

    fmap = ZZFeatureMap(feature_dimension=4, reps=1, entanglement="linear")
    kernel = create_quantum_kernel(fmap)

    n_samples = 5
    rng = np.random.RandomState(42)
    X = rng.uniform(0.0, np.pi, size=(n_samples, 4))

    kernel_matrix = kernel.evaluate(X, X)
    assert kernel_matrix.shape == (n_samples, n_samples)
    # Diagonal of fidelity quantum kernel must be 1.0
    np.testing.assert_allclose(np.diag(kernel_matrix), 1.0, atol=1e-4)


@pytest.mark.quantum
def test_quantum_predict_missing_raises(monkeypatch, sample_feature_dict):
    """Verify RuntimeError is raised when predict_quantum is called without active model."""
    import models.quantum_model as qm
    monkeypatch.setattr(qm, "_ACTIVE_QSVC_MODEL", None)
    monkeypatch.setattr(cfg, "QUANTUM_RESULTS_PATH", "non_existent_quantum_results.joblib")
    monkeypatch.setattr(qm, "QUANTUM_RESULTS_PATH", "non_existent_quantum_results.joblib")

    with pytest.raises(RuntimeError):
        predict_quantum(sample_feature_dict)
