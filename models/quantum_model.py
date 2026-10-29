# URK24CS1111
# QuantumSense - Quantum Support Vector Classifier (QSVC) Module
# Implements Quantum Kernel-Enhanced Classification for Sensor Fault Detection

"""
quantum_model.py - Quantum Support Vector Classifier (QSVC) using Qiskit Machine Learning.
Encodes multi-modal sensor signals into high-dimensional Hilbert space via ZZFeatureMap,
computes quantum kernel fidelity matrices, fits QSVC, and benchmarks against classical SVM.
"""

import os
import sys
import time
from typing import Optional, Dict, List, Tuple, Union, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    QUANTUM_RESULTS_PATH,
    QUANTUM_SCALER_PATH,
    QUANTUM_FEATURE_COUNT,
    QUANTUM_TRAIN_SAMPLES,
    QUANTUM_TEST_SAMPLES,
    ZZ_REPS,
    ZZ_ENTANGLEMENT,
    FEATURE_COLUMNS,
    FAULT_CLASSES,
    CLASS_LABELS,
    RANDOM_SEED,
    TEMP_MEAN,
    TEMP_STD,
    PRESSURE_MEAN,
    PRESSURE_STD,
    VIBRATION_MEAN,
    VIBRATION_STD,
    ensure_dirs,
)
from src.preprocess import prepare_classical_data, prepare_quantum_data
from models.classical_model import train_classical_model, predict_classical

# Qiskit 1.x and Qiskit Machine Learning 0.7+ imports
_QISKIT_IMPORT_ERROR = None
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms import QSVC
except ImportError as err:
    ZZFeatureMap = None
    FidelityQuantumKernel = None
    QSVC = None
    _QISKIT_IMPORT_ERROR = str(err)

# Sampler resolution for Statevector fidelity computation
Sampler = None
if _QISKIT_IMPORT_ERROR is None:
    try:
        from qiskit.primitives import StatevectorSampler as Sampler
    except ImportError:
        try:
            from qiskit.primitives import Sampler
        except ImportError:
            try:
                from qiskit_aer.primitives import Sampler
            except ImportError:
                Sampler = None

# In-memory session cache for active QSVC model and components
_ACTIVE_QSVC_MODEL = None
_ACTIVE_QUANTUM_SCALER = None
_ACTIVE_SELECTED_FEATURES = None
_ACTIVE_QUANTUM_KERNEL = None


def get_quantum_circuit_info(
    feature_dimension: int = QUANTUM_FEATURE_COUNT,
    reps: int = ZZ_REPS,
    entanglement: str = ZZ_ENTANGLEMENT,
) -> dict:
    """
    Construct the ZZFeatureMap quantum circuit and extract structural metrics
    including qubit count, circuit depth, gate counts, and ASCII diagram.

    Args:
        feature_dimension (int): Number of quantum features (qubits)
        reps (int): Number of entangling circuit repetitions
        entanglement (str): Entanglement topology ('linear', 'full', 'circular')

    Returns:
        dict: Circuit architectural details and text diagram.
    """
    if _QISKIT_IMPORT_ERROR is not None:
        raise ImportError(f"Qiskit Machine Learning is unavailable: {_QISKIT_IMPORT_ERROR}")

    feature_map = ZZFeatureMap(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement=entanglement,
    )
    decomposed = feature_map.decompose()

    return {
        "num_qubits": feature_map.num_qubits,
        "circuit_depth": feature_map.depth(),
        "depth": feature_map.depth(),
        "decomposed_depth": decomposed.depth(),
        "parameter_count": feature_map.num_parameters,
        "circuit_diagram": str(decomposed.draw(output="text")),
        "reps": reps,
        "entanglement": entanglement,
        "feature_dimension": feature_dimension,
    }


def _create_quantum_kernel(feature_map):
    """
    Instantiate FidelityQuantumKernel across various Qiskit Machine Learning versions.
    """
    # Strategy 1: Direct instantiation (qiskit-machine-learning 0.7+)
    try:
        return FidelityQuantumKernel(feature_map=feature_map)
    except (TypeError, Exception):
        pass

    # Strategy 2: Explicit ComputeUncompute with Sampler
    try:
        try:
            from qiskit_algorithms.state_fidelities import ComputeUncompute
        except ImportError:
            try:
                from qiskit.algorithms.state_fidelities import ComputeUncompute
            except ImportError:
                from qiskit_machine_learning.state_fidelities import ComputeUncompute

        sampler_inst = Sampler() if Sampler is not None else None
        fidelity = ComputeUncompute(sampler=sampler_inst) if sampler_inst else ComputeUncompute()
        return FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    except Exception:
        pass

    # Strategy 3: Legacy QuantumKernel fallback
    try:
        from qiskit_machine_learning.kernels import QuantumKernel
        return QuantumKernel(feature_map=feature_map)
    except Exception as err:
        raise RuntimeError(f"Unable to initialize quantum kernel with current Qiskit setup: {err}")


create_quantum_kernel = _create_quantum_kernel


def train_quantum_model(
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
    X_train_custom = None,
    y_train_custom = None,
    X_test_custom = None,
    y_test_custom = None,
    feature_names = None,
    class_labels = None,
) -> dict:
    """
    Train a Quantum Support Vector Classifier (QSVC) using a ZZFeatureMap
    and quantum fidelity kernel.

    Pipeline:
      1. Calls prepare_classical_data() for full dataset split (or uses custom split)
      2. Calls prepare_quantum_data() for stratified subsample and angle scaling
      3. Builds 4-qubit ZZFeatureMap quantum circuit
      4. Instantiates FidelityQuantumKernel
      5. Trains QSVC with statevector quantum kernel simulation
      6. Evaluates test set and saves artefacts to disk
      7. Returns dictionary with predictions, timing, and circuit metrics

    Returns:
        dict: Comprehensive results dictionary
    """
    global _ACTIVE_QSVC_MODEL, _ACTIVE_QUANTUM_SCALER, _ACTIVE_SELECTED_FEATURES, _ACTIVE_QUANTUM_KERNEL

    if _QISKIT_IMPORT_ERROR is not None:
        raise ImportError(
            f"Cannot train QSVC model because Qiskit Machine Learning failed to import.\n"
            f"Underlying error: {_QISKIT_IMPORT_ERROR}"
        )

    ensure_dirs()

    n_train = train_samples if train_samples is not None else QUANTUM_TRAIN_SAMPLES
    n_test = test_samples if test_samples is not None else QUANTUM_TEST_SAMPLES

    if X_train_custom is not None and y_train_custom is not None:
        X_train_c = np.asarray(X_train_custom, dtype=float)
        y_train_c = np.asarray(y_train_custom, dtype=int).ravel()
        X_test_c = np.asarray(X_test_custom if X_test_custom is not None else X_train_custom, dtype=float)
        y_test_c = np.asarray(y_test_custom if y_test_custom is not None else y_train_custom, dtype=int).ravel()
    else:
        print("[Quantum QSVC] Step 1/5: Loading base sensor data...")
        classical_data = prepare_classical_data()
        if isinstance(classical_data, (tuple, list)):
            X_train_c, X_test_c, y_train_c, y_test_c = classical_data[:4]
        elif isinstance(classical_data, dict):
            X_train_c = classical_data["X_train"]
            X_test_c = classical_data["X_test"]
            y_train_c = classical_data["y_train"]
            y_test_c = classical_data["y_test"]
        else:
            raise TypeError(f"Unexpected return type from prepare_classical_data: {type(classical_data)}")

    print("[Quantum QSVC] Step 2/5: Preparing quantum subset (angle scaling to [0, pi])...")
    q_data = prepare_quantum_data(
        X_train=X_train_c,
        X_test=X_test_c,
        y_train=y_train_c,
        y_test=y_test_c,
        feature_names=feature_names,
        feature_count=QUANTUM_FEATURE_COUNT,
        train_samples=n_train,
        test_samples=n_test,
    )

    if len(q_data) == 6:
        if hasattr(q_data[4], "transform"):
            X_train_q, X_test_q, y_train_q, y_test_q, quantum_scaler, selected_features = q_data
        else:
            X_train_q, X_test_q, y_train_q, y_test_q, selected_features, quantum_scaler = q_data
    elif len(q_data) == 5:
        X_train_q, X_test_q, y_train_q, y_test_q, quantum_scaler = q_data
        selected_features = FEATURE_COLUMNS[:QUANTUM_FEATURE_COUNT]
    else:
        X_train_q, X_test_q, y_train_q, y_test_q = q_data[:4]
        quantum_scaler = None
        selected_features = FEATURE_COLUMNS[:QUANTUM_FEATURE_COUNT]

    y_train_q = np.asarray(y_train_q, dtype=int).ravel()
    y_test_q = np.asarray(y_test_q, dtype=int).ravel()

    train_samples = len(X_train_q)
    test_samples = len(X_test_q)
    print(f"[Quantum QSVC] Quantum dataset size: {train_samples} train samples, {test_samples} test samples.")
    print(f"[Quantum QSVC] Features selected for quantum encoding: {selected_features}")

    # Build ZZFeatureMap
    print(f"[Quantum QSVC] Step 3/5: Synthesizing ZZFeatureMap ({QUANTUM_FEATURE_COUNT} qubits, reps={ZZ_REPS}, entanglement='{ZZ_ENTANGLEMENT}')...")
    feature_map = ZZFeatureMap(
        feature_dimension=QUANTUM_FEATURE_COUNT,
        reps=ZZ_REPS,
        entanglement=ZZ_ENTANGLEMENT,
    )
    num_qubits = feature_map.num_qubits
    circuit_depth = feature_map.depth()
    feature_map_desc = (
        f"ZZFeatureMap(num_qubits={num_qubits}, reps={ZZ_REPS}, entanglement='{ZZ_ENTANGLEMENT}', depth={circuit_depth})"
    )

    # Instantiate Quantum Kernel
    kernel = _create_quantum_kernel(feature_map)

    # Initialize QSVC
    try:
        qsvc = QSVC(quantum_kernel=kernel)
    except TypeError:
        qsvc = QSVC(kernel=kernel)

    print("=" * 65)
    print(" [NOTE] Evaluating Quantum Kernel Gram Matrix via Statevector Simulation.")
    print(f" Train matrix size: {train_samples}x{train_samples} | Test matrix size: {test_samples}x{train_samples}")
    print(" Simulation may require 15-45 seconds depending on CPU speed...")
    print("=" * 65)

    # Train QSVC model
    start_train = time.time()
    qsvc.fit(X_train_q, y_train_q)
    train_time = time.time() - start_train
    print(f"[Quantum QSVC] Training completed in {train_time:.4f} seconds.")

    # Predict on test set
    print("[Quantum QSVC] Step 4/5: Evaluating quantum kernel predictions on test set...")
    start_pred = time.time()
    y_pred_q = qsvc.predict(X_test_q)
    predict_time = time.time() - start_pred
    print(f"[Quantum QSVC] Test prediction completed in {predict_time:.4f} seconds.")

    # Resolve active class labels dictionary
    active_labels = class_labels if class_labels is not None else FAULT_CLASSES
    target_names = [str(active_labels.get(i, f"Class {i}")) for i in sorted(np.unique(y_test_q))]

    # Calculate metrics
    acc = float(accuracy_score(y_test_q, y_pred_q))
    prec_macro = float(precision_score(y_test_q, y_pred_q, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test_q, y_pred_q, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test_q, y_pred_q, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_test_q, y_pred_q, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test_q, y_pred_q, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test_q, y_pred_q, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test_q, y_pred_q).tolist()
    clf_report = classification_report(
        y_test_q,
        y_pred_q,
        target_names=target_names,
        zero_division=0,
    )

    # Update active session cache
    _ACTIVE_QSVC_MODEL = qsvc
    _ACTIVE_QUANTUM_SCALER = quantum_scaler
    _ACTIVE_SELECTED_FEATURES = selected_features
    _ACTIVE_QUANTUM_KERNEL = kernel

    # Persist artefacts
    print(f"[Quantum QSVC] Step 5/5: Persisting results to: {QUANTUM_RESULTS_PATH}")
    results_to_save = {
        "y_test_q": y_test_q.tolist() if hasattr(y_test_q, "tolist") else list(y_test_q),
        "y_pred_q": y_pred_q.tolist() if hasattr(y_pred_q, "tolist") else list(y_pred_q),
        "accuracy": acc,
        "precision": prec_macro,
        "recall": rec_macro,
        "f1": f1_macro,
        "weighted_precision": prec_weighted,
        "weighted_recall": rec_weighted,
        "weighted_f1": f1_weighted,
        "confusion_matrix": cm,
        "train_time": train_time,
        "predict_time": predict_time,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "num_qubits": num_qubits,
        "circuit_depth": circuit_depth,
        "feature_map_desc": feature_map_desc,
        "classification_report_str": clf_report,
        "selected_features": selected_features,
        "class_labels": active_labels,
        "config": {
            "reps": ZZ_REPS,
            "entanglement": ZZ_ENTANGLEMENT,
            "feature_count": QUANTUM_FEATURE_COUNT,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "seed": RANDOM_SEED,
        },
    }

    # Attempt to pickle the model inside results dict if possible
    try:
        results_with_model = dict(results_to_save)
        results_with_model["model"] = qsvc
        joblib.dump(results_with_model, QUANTUM_RESULTS_PATH)
    except Exception:
        joblib.dump(results_to_save, QUANTUM_RESULTS_PATH)

    if quantum_scaler is not None:
        print(f"[Quantum QSVC] Saving quantum scaler to: {QUANTUM_SCALER_PATH}")
        joblib.dump(quantum_scaler, QUANTUM_SCALER_PATH)

    # Return complete dictionary
    results = {
        "y_test_q": y_test_q,
        "y_pred_q": y_pred_q,
        "accuracy": acc,
        "precision": prec_macro,
        "recall": rec_macro,
        "f1": f1_macro,
        "weighted_precision": prec_weighted,
        "weighted_recall": rec_weighted,
        "weighted_f1": f1_weighted,
        "confusion_matrix": cm,
        "train_time": train_time,
        "predict_time": predict_time,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "num_qubits": num_qubits,
        "circuit_depth": circuit_depth,
        "feature_map_desc": feature_map_desc,
        "classification_report_str": clf_report,
        "selected_features": selected_features,
    }

    return results


def predict_quantum(features: dict) -> dict:
    """
    Perform single-sample inference using the trained Quantum QSVC model.

    Args:
        features (dict): Dictionary mapping feature names to numerical values.

    Returns:
        dict:
            - predicted_class: Integer class ID (0..4)
            - predicted_label: Human-readable fault class name
            - confidence: Confidence measure (or 1.0 for hard margin)
            - features_used: List of feature names encoded into qubits
            - num_qubits: Number of qubits used in the circuit

    Raises:
        RuntimeError: If quantum model is not loaded in memory or saved artefacts.
    """
    global _ACTIVE_QSVC_MODEL, _ACTIVE_QUANTUM_SCALER, _ACTIVE_SELECTED_FEATURES

    model = _ACTIVE_QSVC_MODEL
    scaler = _ACTIVE_QUANTUM_SCALER
    selected_features = _ACTIVE_SELECTED_FEATURES

    # Attempt recovery from stored artefacts if in-memory cache is empty
    if model is None and os.path.exists(QUANTUM_RESULTS_PATH):
        try:
            saved_results = joblib.load(QUANTUM_RESULTS_PATH)
            if isinstance(saved_results, dict):
                model = saved_results.get("model")
                if selected_features is None:
                    selected_features = saved_results.get("selected_features")
        except Exception:
            model = None

    if scaler is None and os.path.exists(QUANTUM_SCALER_PATH):
        try:
            scaler = joblib.load(QUANTUM_SCALER_PATH)
        except Exception:
            scaler = None

    if model is None:
        raise RuntimeError(
            "Quantum QSVC model is not available in memory or persisted storage!\n"
            "Because quantum circuit simulators involve non-serializable Qiskit primitive sessions, "
            "the quantum model requires an active training session in memory.\n"
            "Please execute train_quantum_model() before attempting inference. "
            "(Note: QuantumSense deliberately does not fall back to classical models for quantum requests)."
        )

    if selected_features is None:
        selected_features = ["temp_mean", "temp_std", "pres_mean", "vib_mean"]

    # Build input feature vector
    row = []
    for col in selected_features:
        if col in features:
            val = float(features[col])
        else:
            if col.startswith("temp"):
                val = TEMP_MEAN if "mean" in col else (TEMP_STD if "std" in col else 0.0)
            elif col.startswith("pres"):
                val = PRESSURE_MEAN if "mean" in col else (PRESSURE_STD if "std" in col else 0.0)
            elif col.startswith("vib"):
                val = VIBRATION_MEAN if "mean" in col else (VIBRATION_STD if "std" in col else 0.0)
            else:
                val = 0.0
        row.append(val)

    X_input = np.array([row])
    if scaler is not None:
        X_scaled = scaler.transform(X_input)
    else:
        # Scale to [0, pi]
        X_scaled = np.clip(X_input, 0.0, np.pi)

    pred_class = int(model.predict(X_scaled)[0])
    pred_label = FAULT_CLASSES.get(pred_class, f"Class {pred_class}")

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_scaled)[0]
            confidence = float(np.max(probs))
        except Exception:
            confidence = 1.0
    elif hasattr(model, "decision_function"):
        try:
            dfunc = model.decision_function(X_scaled)[0]
            # Softmax on decision function for indicative confidence
            exp_d = np.exp(dfunc - np.max(dfunc))
            confidence = float(np.max(exp_d / np.sum(exp_d)))
        except Exception:
            confidence = 1.0

    return {
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "confidence": confidence,
        "features_used": selected_features,
        "num_qubits": len(selected_features),
    }


if __name__ == "__main__":
    print("=" * 70)
    print(" QuantumSense - Quantum Support Vector Classifier (QSVC) Training")
    print("=" * 70)

    # Display Circuit Architecture
    try:
        print("\nConstructing ZZFeatureMap Quantum Circuit:")
        circuit_info = get_quantum_circuit_info()
        print(f"Qubits               : {circuit_info['num_qubits']}")
        print(f"Repetitions (reps)   : {circuit_info['reps']}")
        print(f"Entanglement Topology: {circuit_info['entanglement']}")
        print(f"Circuit Depth        : {circuit_info['circuit_depth']}")
        print(f"Decomposed Depth     : {circuit_info['decomposed_depth']}")
        print(f"Parameters           : {circuit_info['parameter_count']}")
        print("\nDecomposed Quantum Circuit Diagram:")
        print(circuit_info["circuit_diagram"])
    except Exception as e:
        print(f"[Warning] Could not render circuit info: {e}")

    # Train Quantum Model
    try:
        print("\n" + "=" * 50)
        print("Starting Quantum Model Training & Evaluation")
        print("=" * 50)
        q_results = train_quantum_model()

        print("\n" + "=" * 50)
        print("Quantum QSVC Performance Summary:")
        print("=" * 50)
        print(f"Accuracy         : {q_results['accuracy']:.4f} ({q_results['accuracy'] * 100:.2f}%)")
        print(f"Precision (macro): {q_results['precision']:.4f}")
        print(f"Recall (macro)   : {q_results['recall']:.4f}")
        print(f"F1-Score (macro) : {q_results['f1']:.4f}")
        print(f"Training Time    : {q_results['train_time']:.4f} s")
        print(f"Test Infer Time  : {q_results['predict_time']:.4f} s")
        print(f"Training Samples : {q_results['train_samples']}")
        print(f"Testing Samples  : {q_results['test_samples']}")
        print("\nClassification Report:\n")
        print(q_results["classification_report_str"])

        print("\nTesting single-sample quantum inference pipeline...")
        test_features = {
            "temp_mean": 75.0,
            "temp_std": 2.0,
            "pres_mean": 101.3,
            "vib_mean": 0.5,
        }
        pred = predict_quantum(test_features)
        print(f"Prediction: {pred['predicted_label']} (Class {pred['predicted_class']})")
        print(f"Confidence: {pred['confidence']:.2%}")
        print(f"Features Encoded: {pred['features_used']}")

    except Exception as e:
        print(f"[ERROR] Failed during quantum model execution: {e}")
        import traceback
        traceback.print_exc()
