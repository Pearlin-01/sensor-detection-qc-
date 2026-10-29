# URK24CS1111
# QuantumSense - Classical SVM Model Module
# Implements Support Vector Machine classification for sensor fault detection

"""
classical_model.py - Classical Machine Learning baseline using Scikit-Learn SVC.
Trains an RBF-kernel SVM classifier on extracted statistical sensor features,
evaluates performance metrics, and provides single-sample inference capabilities.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
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
    CLASSICAL_MODEL_PATH,
    CLASSICAL_SCALER_PATH,
    RESULTS_DIR,
    FEATURE_COLUMNS,
    FAULT_CLASSES,
    CLASS_LABELS,
    RANDOM_SEED,
    SVM_KERNEL,
    SVM_C,
    SVM_GAMMA,
    TEMP_MEAN,
    TEMP_STD,
    PRESSURE_MEAN,
    PRESSURE_STD,
    VIBRATION_MEAN,
    VIBRATION_STD,
    ensure_dirs,
)
from src.preprocess import prepare_classical_data


def train_classical_model(
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
    scaler=None,
    kernel=SVM_KERNEL,
    C=SVM_C,
    gamma=SVM_GAMMA,
    feature_names=None,
    class_labels=None,
) -> dict:
    """
    Train a classical Support Vector Classifier (SVC) with RBF kernel
    for multi-class sensor fault detection.
    """
    ensure_dirs()

    if X_train is None or y_train is None:
        print("[Classical SVM] Preparing training and testing datasets...")
        data = prepare_classical_data()

        # Unpack data flexibly (supports tuple or dictionary return formats)
        if isinstance(data, (tuple, list)):
            X_train, X_test, y_train, y_test, scaler = data[:5]
        elif isinstance(data, dict):
            X_train = data["X_train"]
            X_test = data["X_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]
            scaler = data["scaler"]
        else:
            raise TypeError(f"Unexpected return type from prepare_classical_data: {type(data)}")
    else:
        if X_test is None or y_test is None:
            X_test = X_train
            y_test = y_train

    # Ensure y arrays are 1D integer arrays
    y_train = np.asarray(y_train, dtype=int).ravel()
    y_test = np.asarray(y_test, dtype=int).ravel()

    train_samples = len(X_train)
    test_samples = len(X_test)
    print(f"[Classical SVM] Dataset split: {train_samples} training samples, {test_samples} testing samples.")

    # Initialize Support Vector Classifier
    model = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        random_state=RANDOM_SEED,
        probability=True,
    )

    # Train model and measure training wall time
    print(f"[Classical SVM] Fitting SVC(kernel='{SVM_KERNEL}', C={SVM_C}, gamma='{SVM_GAMMA}')...")
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"[Classical SVM] Model training completed in {train_time:.4f} seconds.")

    # Predict on test set and measure inference time
    start_pred = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_pred
    print(f"[Classical SVM] Inference on {test_samples} test samples completed in {predict_time:.4f} seconds.")

    # Resolve active class labels dictionary
    active_labels = class_labels if class_labels is not None else FAULT_CLASSES
    unique_classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    target_names = [str(active_labels.get(i, f"Class {i}")) for i in sorted(np.unique(y_test))]

    # Compute comprehensive evaluation metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )

    # Save model, scaler and metadata artefacts
    print(f"[Classical SVM] Saving model to: {CLASSICAL_MODEL_PATH}")
    joblib.dump(model, CLASSICAL_MODEL_PATH)
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_train)
    print(f"[Classical SVM] Saving scaler to: {CLASSICAL_SCALER_PATH}")
    joblib.dump(scaler, CLASSICAL_SCALER_PATH)

    active_features = feature_names if feature_names is not None else FEATURE_COLUMNS
    meta_path = os.path.join(RESULTS_DIR, "classical_metadata.joblib")
    joblib.dump({"features": active_features, "class_labels": active_labels}, meta_path)

    results = {
        "model": model,
        "scaler": scaler,
        "y_test": y_test,
        "y_pred": y_pred,
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
        "classification_report_str": clf_report,
        "features": active_features,
        "class_labels": active_labels,
    }

    return results


def predict_classical(features: dict) -> dict:
    """
    Perform single-sample inference using the trained classical SVM model.

    Args:
        features (dict): Dictionary mapping feature column names to numerical values.
                         e.g. {"temp_mean": 75.2, "temp_std": 2.1, ...}

    Returns:
        dict:
            - predicted_class: Integer class ID (0..4)
            - predicted_label: Human-readable fault name (e.g. 'Normal', 'Drift')
            - confidence: Model confidence probability score (0.0 to 1.0)
            - probabilities: Dictionary of class label -> probability
            - features_used: Features aligned with model inputs
    """
    if not os.path.exists(CLASSICAL_MODEL_PATH) or not os.path.exists(CLASSICAL_SCALER_PATH):
        raise FileNotFoundError(
            f"Classical model or scaler artefacts missing!\n"
            f"Model path: '{CLASSICAL_MODEL_PATH}' (exists: {os.path.exists(CLASSICAL_MODEL_PATH)})\n"
            f"Scaler path: '{CLASSICAL_SCALER_PATH}' (exists: {os.path.exists(CLASSICAL_SCALER_PATH)})\n"
            f"Please run train_classical_model() before attempting inference."
        )

    model = joblib.load(CLASSICAL_MODEL_PATH)
    scaler = joblib.load(CLASSICAL_SCALER_PATH)

    meta_path = os.path.join(RESULTS_DIR, "classical_metadata.joblib")
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
        expected_cols = meta.get("features", FEATURE_COLUMNS)
        active_classes = meta.get("class_labels", FAULT_CLASSES)
    else:
        expected_cols = FEATURE_COLUMNS
        active_classes = FAULT_CLASSES

    # Build input feature vector aligning with expected features
    row = {}
    for col in expected_cols:
        if col in features:
            try:
                row[col] = float(features[col])
            except (ValueError, TypeError):
                row[col] = 0.0
        else:
            # Impute sensor nominal values if recognized, else 0.0
            if col.startswith("temp"):
                row[col] = TEMP_MEAN if "mean" in col else (TEMP_STD if "std" in col else 0.0)
            elif col.startswith("pres"):
                row[col] = PRESSURE_MEAN if "mean" in col else (PRESSURE_STD if "std" in col else 0.0)
            elif col.startswith("vib"):
                row[col] = VIBRATION_MEAN if "mean" in col else (VIBRATION_STD if "std" in col else 0.0)
            else:
                row[col] = 0.0

    df_input = pd.DataFrame([row], columns=expected_cols)
    scaled_input = scaler.transform(df_input)

    pred_class = int(model.predict(scaled_input)[0])
    pred_label = active_classes.get(pred_class, f"Class {pred_class}")

    probabilities = {}
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled_input)[0]
        confidence = float(np.max(probs))
        # Map classes to their probabilities
        classes = getattr(model, "classes_", range(len(probs)))
        for idx, cls in enumerate(classes):
            label = active_classes.get(int(cls), f"Class {cls}")
            probabilities[label] = float(probs[idx])

    return {
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "features_used": row,
    }


if __name__ == "__main__":
    print("=" * 70)
    print(" QuantumSense - Classical SVM Sensor Fault Detection Model")
    print("=" * 70)

    try:
        results = train_classical_model()
        print("\n" + "=" * 50)
        print("Classical SVM Performance Summary:")
        print("=" * 50)
        print(f"Accuracy         : {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
        print(f"Precision (macro): {results['precision']:.4f}")
        print(f"Recall (macro)   : {results['recall']:.4f}")
        print(f"F1-Score (macro) : {results['f1']:.4f}")
        print(f"Training Time    : {results['train_time']:.4f} s")
        print(f"Test Infer Time  : {results['predict_time']:.4f} s")
        print(f"Training Samples : {results['train_samples']}")
        print(f"Testing Samples  : {results['test_samples']}")
        print("\nClassification Report:\n")
        print(results["classification_report_str"])

        print("\nTesting single-sample prediction pipeline...")
        nominal_features = {
            "temp_mean": TEMP_MEAN,
            "temp_std": TEMP_STD,
            "temp_min": TEMP_MEAN - 2.0,
            "temp_max": TEMP_MEAN + 2.0,
            "temp_slope": 0.01,
            "pres_mean": PRESSURE_MEAN,
            "pres_std": PRESSURE_STD,
            "pres_min": PRESSURE_MEAN - 1.5,
            "pres_max": PRESSURE_MEAN + 1.5,
            "pres_slope": 0.0,
            "vib_mean": VIBRATION_MEAN,
            "vib_std": VIBRATION_STD,
            "vib_min": VIBRATION_MEAN - 0.1,
            "vib_max": VIBRATION_MEAN + 0.1,
            "vib_slope": 0.0,
        }
        test_pred = predict_classical(nominal_features)
        print(f"Sample Input -> Predicted Class: {test_pred['predicted_class']} ({test_pred['predicted_label']})")
        print(f"Confidence: {test_pred['confidence']:.2%}")
        print("Class Probabilities:")
        for lbl, prob in test_pred["probabilities"].items():
            print(f"  - {lbl:10s}: {prob:.4f}")

    except Exception as e:
        print(f"[ERROR] Failed during classical model training/inference: {e}")
        import traceback
        traceback.print_exc()
