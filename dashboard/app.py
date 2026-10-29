# URK24CS1111
"""
QuantumSense - Enhanced Web Dashboard Backend
Flask REST API and industrial monitoring application for quantum-enhanced sensor fault detection.
Supports both synthetic simulated telemetry and custom user-uploaded CSV datasets,
sliding-window feature engineering, classical SVM and quantum QSVC training,
dataset analytics, batch fault prediction, maintenance advisories, and experiment history.
"""

import os
import sys
import json
import time
import random
import threading
from datetime import datetime
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

# Ensure project root is in sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    UPLOADS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SENSOR_DATA_CSV,
    CUSTOM_DATA_CSV,
    ACTIVE_DATASET_CONFIG,
    CLEANED_DATA_CSV,
    BATCH_PREDICTIONS_CSV,
    MAX_UPLOAD_SIZE_BYTES,
    CLASSICAL_MODEL_PATH,
    CLASSICAL_SCALER_PATH,
    QUANTUM_RESULTS_PATH,
    QUANTUM_SCALER_PATH,
    METRICS_JSON,
    PREDICTIONS_CSV,
    EXPERIMENT_HISTORY_JSON,
    COMPARISON_PNG,
    CM_CLASSICAL_PNG,
    CM_QUANTUM_PNG,
    RANDOM_SEED,
    SAMPLES_PER_CLASS,
    WINDOW_SIZE,
    TEMP_MEAN,
    TEMP_STD,
    PRESSURE_MEAN,
    PRESSURE_STD,
    VIBRATION_MEAN,
    VIBRATION_STD,
    FAULT_CLASSES,
    NUM_CLASSES,
    CLASS_LABELS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LABEL_COLUMN,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    ensure_dirs,
)

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

from models.classical_model import train_classical_model, predict_classical
from models.quantum_model import train_quantum_model, predict_quantum
from src.evaluate import evaluate_models

try:
    from src.generate_data import generate_single_sample as ext_generate_single_sample
except (ImportError, ModuleNotFoundError):
    ext_generate_single_sample = None

import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, Response

# Ensure required directories exist
ensure_dirs()

template_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_BYTES

# ---------------------------------------------------------------------------
# In-memory history stores & Background Training State
# ---------------------------------------------------------------------------
MAX_READINGS_HISTORY = 100
MAX_PREDICTIONS_HISTORY = 50

recent_sensor_readings = []
recent_predictions = []

training_state_lock = threading.Lock()
training_state = {
    "status": "idle",       # "idle", "running", "completed", "failed"
    "progress": 0,          # 0 to 100
    "stage": "Ready",
    "logs": [],
    "results": None,
    "error": None,
    "cancel_requested": False,
    "start_time": None,
    "end_time": None,
}


# ---------------------------------------------------------------------------
# Maintenance Recommendations Engine
# ---------------------------------------------------------------------------
MAINTENANCE_ADVISORIES = {
    0: {
        "pattern": "Normal operating parameters within standard nominal statistical variance.",
        "inspection_steps": [
            "Perform scheduled periodic telemetry calibration check.",
            "Verify network latency and sensor DAQ heartbeat consistency.",
            "Log operational hours into preventive maintenance log.",
        ],
        "recommended_action": "No corrective action required. Continue routine telemetry monitoring.",
        "severity": "NORMAL",
        "severity_color": "var(--success)",
    },
    1: {
        "pattern": "Gradual monotonic parameter drift detected exceeding calibration boundaries.",
        "inspection_steps": [
            "Inspect physical sensor probe for zero-point drift or thermal bias.",
            "Verify probe cleanliness and check for particulate/chemical fouling.",
            "Perform multi-point calibration recalibration against certified reference standard.",
        ],
        "recommended_action": "Schedule sensor recalibration at the next available maintenance window.",
        "severity": "MEDIUM",
        "severity_color": "var(--warning)",
    },
    2: {
        "pattern": "Signal variance frozen or near-zero, indicating sensor hardware lockup or DAQ failure.",
        "inspection_steps": [
            "Inspect sensor power supply voltage and 4-20mA / I2C signal wiring integrity.",
            "Examine ADC input channel and analog front-end conditioning circuitry.",
            "Check for physical mechanical seizure of sensor transducer or diaphragm.",
        ],
        "recommended_action": "Immediate technician dispatch required: reset or replace frozen sensor hardware.",
        "severity": "HIGH",
        "severity_color": "#ff9100",
    },
    3: {
        "pattern": "Excessive random variance (3x-5x nominal), indicating electromagnetic interference or mechanical vibration.",
        "inspection_steps": [
            "Verify cable shielding and chassis grounding loop connections.",
            "Inspect proximity to Variable Frequency Drives (VFD), motors, or high-current conduits.",
            "Inspect mechanical sensor mounting bracket for looseness or resonant vibration.",
        ],
        "recommended_action": "Inspect grounding shielding and secure loose mechanical mounting fasteners.",
        "severity": "MEDIUM-HIGH",
        "severity_color": "#ff6d00",
    },
    4: {
        "pattern": "Transient extreme impulses / spike excursions exceeding physical safety margins.",
        "inspection_steps": [
            "Inspect circuit connectors for intermittent contact resistance or loose terminals.",
            "Verify hydraulic/pneumatic line for sudden water hammer or pressure relief surge.",
            "Examine power supply rail for transient voltage spikes or inductive load collapse.",
        ],
        "recommended_action": "Urgent alert: Inspect electrical connectors and investigate transient hydraulic/mechanical shock.",
        "severity": "CRITICAL",
        "severity_color": "var(--danger)",
    },
}


def get_maintenance_advisory(fault_class_id: int, fault_label: str) -> dict:
    """Return structured engineering maintenance recommendations for a fault class."""
    adv = MAINTENANCE_ADVISORIES.get(fault_class_id, {
        "pattern": f"Anomalous pattern identified for {fault_label}.",
        "inspection_steps": [
            "Inspect physical sensor wiring and verify terminal connections.",
            "Review historical telemetry trends leading up to anomaly onset.",
            "Cross-validate reading against redundant secondary instrument.",
        ],
        "recommended_action": "Investigate sensor telemetry and verify operational parameters.",
        "severity": "WARNING",
        "severity_color": "var(--warning)",
    })
    adv_copy = dict(adv)
    adv_copy["category"] = fault_label
    adv_copy["disclaimer"] = "Advisory guidance based on automated statistical anomaly detection. Physical verification by certified engineering personnel required prior to equipment overhaul."
    return adv_copy


# ---------------------------------------------------------------------------
# CORS Header Hook
# ---------------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


# ---------------------------------------------------------------------------
# Helper: Internal Feature Generator (Fallback / Self-contained)
# ---------------------------------------------------------------------------
def compute_window_features(temp_series, pres_series, vib_series):
    """Calculate 15 statistical features from time series windows."""
    t = np.arange(len(temp_series))
    
    def calc_stats(series):
        mean = float(np.mean(series))
        std = float(np.std(series))
        s_min = float(np.min(series))
        s_max = float(np.max(series))
        if len(series) > 1:
            slope = float(np.polyfit(t, series, 1)[0])
        else:
            slope = 0.0
        return mean, std, s_min, s_max, slope

    t_mean, t_std, t_min, t_max, t_slope = calc_stats(temp_series)
    p_mean, p_std, p_min, p_max, p_slope = calc_stats(pres_series)
    v_mean, v_std, v_min, v_max, v_slope = calc_stats(vib_series)

    features = {
        "temp_mean": round(t_mean, 4),
        "temp_std": round(t_std, 4),
        "temp_min": round(t_min, 4),
        "temp_max": round(t_max, 4),
        "temp_slope": round(t_slope, 4),
        "pres_mean": round(p_mean, 4),
        "pres_std": round(p_std, 4),
        "pres_min": round(p_min, 4),
        "pres_max": round(p_max, 4),
        "pres_slope": round(p_slope, 4),
        "vib_mean": round(v_mean, 4),
        "vib_std": round(v_std, 4),
        "vib_min": round(v_min, 4),
        "vib_max": round(v_max, 4),
        "vib_slope": round(v_slope, 4),
    }
    return features


def simulate_window(fault_class=None):
    """Simulate a time window of sensor readings with an optional fault injected."""
    if fault_class is None:
        fault_class = random.choice([0, 1, 2, 3, 4])
    
    fault_label = FAULT_CLASSES.get(fault_class, "Normal")
    n = WINDOW_SIZE
    t = np.arange(n)

    # Base nominal signals
    temp = np.random.normal(TEMP_MEAN, TEMP_STD, n)
    pres = np.random.normal(PRESSURE_MEAN, PRESSURE_STD, n)
    vib = np.random.normal(VIBRATION_MEAN, VIBRATION_STD, n)

    if fault_class == 1:
        # Drift: slow linear ramp
        drift_rate = random.choice([0.15, -0.15])
        temp += drift_rate * t
        pres += (drift_rate * 0.5) * t
    elif fault_class == 2:
        # Stuck: freeze at constant value
        stuck_idx = random.randint(5, 15)
        temp[stuck_idx:] = temp[stuck_idx]
        pres[stuck_idx:] = pres[stuck_idx]
        vib[stuck_idx:] = 0.01
    elif fault_class == 3:
        # Noisy: excessive random variance
        temp += np.random.normal(0, TEMP_STD * 3.5, n)
        pres += np.random.normal(0, PRESSURE_STD * 3.5, n)
        vib += np.random.normal(0, VIBRATION_STD * 4.0, n)
    elif fault_class == 4:
        # Spike: sharp transient jump
        spike_pos = random.randint(15, 35)
        temp[spike_pos:spike_pos+3] += random.choice([15.0, -15.0])
        pres[spike_pos:spike_pos+3] += random.choice([10.0, -10.0])
        vib[spike_pos:spike_pos+3] += random.choice([1.2, 1.8])

    vib = np.clip(vib, 0.0, None)
    features = compute_window_features(temp, pres, vib)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "simulated_class": fault_class,
        "simulated_label": fault_label,
        "temperature": round(float(temp[-1]), 2),
        "pressure": round(float(pres[-1]), 2),
        "vibration": round(float(vib[-1]), 3),
        "series": {
            "temperature": [round(float(x), 2) for x in temp[-30:]],
            "pressure": [round(float(x), 2) for x in pres[-30:]],
            "vibration": [round(float(x), 3) for x in vib[-30:]],
        },
        "features": features,
        "is_simulated": True
    }


def call_predict_classical(features_dict):
    """Execute classical prediction using module or loaded joblib model."""
    return predict_classical(features_dict)


def call_predict_quantum(features_dict):
    """Execute quantum prediction using module or loaded results."""
    return predict_quantum(features_dict)


# ---------------------------------------------------------------------------
# Existing Core Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the single-page application dashboard."""
    return render_template("index.html")


@app.route("/results/<path:filename>")
def serve_result_file(filename):
    """Serve results files such as confusion matrix and comparison images."""
    file_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": f"Result file '{filename}' not found."}), 404
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/api/status", methods=["GET"])
def get_status():
    """Return system readiness, dataset statistics, and model availability."""
    active_info = get_active_dataset_info()
    dataset_exists = os.path.exists(active_info.get("filepath", SENSOR_DATA_CSV))
    total_samples = active_info.get("total_rows", 0)
    normal_count = 0
    faulty_count = 0

    if dataset_exists:
        try:
            df = pd.read_csv(active_info["filepath"])
            total_samples = int(len(df))
            tgt = active_info.get("target_column", TARGET_COLUMN)
            if tgt in df.columns:
                normal_count = int((df[tgt] == 0).sum())
                faulty_count = int((df[tgt] != 0).sum())
        except Exception:
            total_samples = SAMPLES_PER_CLASS * NUM_CLASSES
            normal_count = SAMPLES_PER_CLASS
            faulty_count = SAMPLES_PER_CLASS * (NUM_CLASSES - 1)
    
    classical_available = os.path.exists(CLASSICAL_MODEL_PATH)
    quantum_available = os.path.exists(QUANTUM_RESULTS_PATH)
    metrics_available = os.path.exists(METRICS_JSON)

    # Get active model feature list
    active_features = active_info.get("feature_columns", FEATURE_COLUMNS)
    meta_path = os.path.join(RESULTS_DIR, "classical_metadata.joblib")
    if os.path.exists(meta_path):
        try:
            cm = joblib.load(meta_path)
            active_features = cm.get("features", active_features)
        except Exception:
            pass

    return jsonify({
        "success": True,
        "dataset_exists": dataset_exists,
        "dataset_path": active_info.get("filepath", SENSOR_DATA_CSV),
        "dataset_name": active_info.get("dataset_name", "Synthetic Telemetry"),
        "dataset_source": active_info.get("source", "synthetic"),
        "is_custom": active_info.get("is_custom", False),
        "total_samples": total_samples,
        "normal_count": normal_count,
        "faulty_count": faulty_count,
        "active_features": active_features,
        "models": {
            "classical_svm": {
                "available": classical_available,
                "path": CLASSICAL_MODEL_PATH
            },
            "quantum_qsvc": {
                "available": quantum_available,
                "path": QUANTUM_RESULTS_PATH
            }
        },
        "metrics_available": metrics_available,
        "fault_classes": active_info.get("fault_classes", FAULT_CLASSES),
        "recent_predictions_count": len(recent_predictions),
        "recent_readings_count": len(recent_sensor_readings),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Load and return METRICS_JSON content or structured error."""
    if not os.path.exists(METRICS_JSON):
        return jsonify({
            "success": False,
            "available": False,
            "error": "Metrics file not found. Models have not been trained yet."
        }), 404

    try:
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        return jsonify({
            "success": True,
            "available": True,
            "metrics": metrics_data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "available": False,
            "error": f"Error loading metrics: {str(e)}"
        }), 500


@app.route("/api/evaluation", methods=["GET"])
def get_evaluation():
    """Return full evaluation metrics and metadata."""
    if not os.path.exists(METRICS_JSON):
        return jsonify({
            "success": False,
            "error": "Evaluation metrics not available. Please train models first."
        }), 404

    try:
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        # Add image availability flags
        eval_data["images"] = {
            "model_comparison": os.path.exists(COMPARISON_PNG),
            "cm_classical": os.path.exists(CM_CLASSICAL_PNG),
            "cm_quantum": os.path.exists(CM_QUANTUM_PNG)
        }
        return jsonify({
            "success": True,
            "evaluation": eval_data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to retrieve evaluation: {str(e)}"
        }), 500


@app.route("/api/generate", methods=["POST"])
def generate_sample():
    """
    Generate a new simulated sensor sample.
    Optional JSON payload: {"fault_class": int}
    """
    try:
        req_data = request.get_json(silent=True) or {}
        requested_fault = req_data.get("fault_class")
        if requested_fault is not None:
            try:
                requested_fault = int(requested_fault)
            except ValueError:
                requested_fault = None

        sample = None
        if ext_generate_single_sample is not None:
            try:
                ext_res = ext_generate_single_sample(fault_type=requested_fault)
                if isinstance(ext_res, dict):
                    sample = ext_res
            except Exception:
                sample = None

        if sample is None:
            sample = simulate_window(fault_class=requested_fault)

        recent_sensor_readings.append(sample)
        if len(recent_sensor_readings) > MAX_READINGS_HISTORY:
            recent_sensor_readings.pop(0)

        return jsonify({
            "success": True,
            "sample": sample
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to generate sensor sample: {str(e)}"
        }), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run fault detection on provided sensor features.
    Expected JSON: {"features": {...}, "model_type": "classical" | "quantum"}
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Missing JSON request body."}), 400

    features = data.get("features")
    if not features or not isinstance(features, dict):
        if recent_sensor_readings:
            features = recent_sensor_readings[-1]["features"]
        else:
            return jsonify({
                "success": False,
                "error": "No feature dictionary provided and no prior readings available."
            }), 400

    model_type = data.get("model_type", "classical").strip().lower()

    if model_type not in ["classical", "quantum"]:
        return jsonify({
            "success": False,
            "error": f"Invalid model_type '{model_type}'. Expected 'classical' or 'quantum'."
        }), 400

    try:
        if model_type == "classical":
            pred = call_predict_classical(features)
            model_name = "Classical SVM"
        else:
            try:
                pred = call_predict_quantum(features)
                model_name = "Quantum QSVC"
            except RuntimeError as re:
                return jsonify({
                    "success": False,
                    "error": str(re),
                    "model_used": "Quantum QSVC"
                }), 500

        pred_class = int(pred.get("predicted_class", pred.get("class", 0)))
        pred_label = str(pred.get("predicted_label", pred.get("label", FAULT_CLASSES.get(pred_class, "Unknown"))))
        confidence = float(pred.get("confidence", 0.95))

        # Maintenance recommendation
        advisory = get_maintenance_advisory(pred_class, pred_label)

        prediction_record = {
            "id": len(recent_predictions) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": model_name,
            "model_type": model_type,
            "class": pred_class,
            "label": pred_label,
            "confidence": round(confidence * 100, 2) if confidence <= 1.0 else round(confidence, 2),
            "is_fault": pred_class != 0,
            "features": features,
            "advisory": advisory,
        }

        recent_predictions.append(prediction_record)
        if len(recent_predictions) > MAX_PREDICTIONS_HISTORY:
            recent_predictions.pop(0)

        return jsonify({
            "success": True,
            "model_used": model_name,
            "model_type": model_type,
            "prediction": {
                "class": pred_class,
                "label": pred_label,
                "confidence": round(confidence * 100, 2) if confidence <= 1.0 else round(confidence, 2)
            },
            "advisory": advisory,
            "timestamp": prediction_record["timestamp"]
        })

    except FileNotFoundError as fe:
        return jsonify({
            "success": False,
            "error": f"Model file error: {str(fe)}. Please train the models first."
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500


@app.route("/api/sensor_history", methods=["GET"])
def get_sensor_history():
    """Return the recent list of sensor readings."""
    limit = request.args.get("limit", default=30, type=int)
    history = recent_sensor_readings[-limit:] if recent_sensor_readings else []
    return jsonify({
        "success": True,
        "count": len(history),
        "history": history
    })


@app.route("/api/prediction_history", methods=["GET"])
def get_prediction_history():
    """Return recent fault prediction history records."""
    limit = request.args.get("limit", default=20, type=int)
    history = recent_predictions[-limit:] if recent_predictions else []
    return jsonify({
        "success": True,
        "count": len(history),
        "history": list(reversed(history))
    })


# ---------------------------------------------------------------------------
# NEW: Custom Dataset Lab Routes
# ---------------------------------------------------------------------------

@app.route("/api/dataset/upload", methods=["POST"])
def upload_dataset():
    """
    Handle CSV upload for Custom Dataset Lab.
    Validates file type (.csv), file size limit, and returns inspection preview.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in upload request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected for upload."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"success": False, "error": "Invalid file type. Only CSV files are supported."}), 400

    ensure_dirs()
    filename = secure_filename(file.filename)
    if not filename:
        filename = f"upload_{int(time.time())}.csv"
    
    save_path = os.path.join(UPLOADS_DIR, filename)
    file.save(save_path)

    try:
        report = inspect_csv(save_path)
        report["uploaded_filename"] = filename
        return jsonify(report)
    except Exception as e:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception:
                pass
        return jsonify({"success": False, "error": f"Failed to parse uploaded CSV: {str(e)}"}), 400


@app.route("/api/dataset/process", methods=["POST"])
def process_custom_dataset():
    """
    Process custom uploaded dataset:
    Mode A (Precomputed Features) or Mode B (Raw Time-Series sliding window).
    Applies cleaning, encoding, and registers as active dataset.
    """
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Missing configuration payload."}), 400

    filename = data.get("filename")
    if not filename:
        return jsonify({"success": False, "error": "Missing filename parameter."}), 400

    filepath = os.path.join(UPLOADS_DIR, secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({"success": False, "error": f"Uploaded file '{filename}' not found on server."}), 404

    mode = data.get("mode", "precomputed")  # 'precomputed' (Mode A) or 'window' (Mode B)
    feature_cols = data.get("feature_columns", [])
    target_col = data.get("target_column") or None
    timestamp_col = data.get("timestamp_column") or None
    window_size = int(data.get("window_size", 50))
    missing_strat = data.get("missing_strategy", "mean")
    drop_dups = bool(data.get("drop_duplicates", False))
    handle_outliers = bool(data.get("handle_outliers", False))

    if not feature_cols:
        return jsonify({"success": False, "error": "At least one sensor feature column must be selected."}), 400

    try:
        raw_df = pd.read_csv(filepath)

        if mode == "window":
            # Mode B: Raw Time Series Sliding Window
            processed_df = extract_window_features(
                df=raw_df,
                sensor_columns=feature_cols,
                window_size=window_size,
                target_col=target_col,
                timestamp_col=timestamp_col,
            )
            # Extracted features are column_mean, column_std, etc.
            new_feature_cols = [c for c in processed_df.columns if c not in ["fault_label", target_col]]
            clean_df, label_mapping, clean_report = clean_and_preprocess(
                df=processed_df,
                feature_columns=new_feature_cols,
                target_column="fault_label" if "fault_label" in processed_df.columns else None,
                missing_strategy=missing_strat,
                drop_duplicates=drop_dups,
                handle_outliers=handle_outliers,
            )
            actual_features = new_feature_cols
        else:
            # Mode A: Precomputed Features
            clean_df, label_mapping, clean_report = clean_and_preprocess(
                df=raw_df,
                feature_columns=feature_cols,
                target_column=target_col,
                missing_strategy=missing_strat,
                drop_duplicates=drop_dups,
                handle_outliers=handle_outliers,
            )
            actual_features = feature_cols

        # Save as custom dataset and activate
        meta = {
            "dataset_name": f"Custom Dataset ({filename})",
            "source": "custom",
            "filename": filename,
            "mode": mode,
            "feature_columns": actual_features,
            "target_column": "fault_class" if clean_report["has_labels"] else None,
            "label_column": "fault_label" if clean_report["has_labels"] else None,
            "fault_classes": label_mapping,
            "cleaning_report": clean_report,
        }

        saved_path = save_custom_dataset(clean_df, meta)

        return jsonify({
            "success": True,
            "dataset_path": saved_path,
            "total_samples": len(clean_df),
            "feature_count": len(actual_features),
            "feature_columns": actual_features,
            "has_labels": clean_report["has_labels"],
            "classes": label_mapping,
            "cleaning_report": clean_report,
            "preview": clean_df.head(10).replace({np.nan: None}).to_dict(orient="records"),
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Dataset processing error: {str(e)}"}), 500


@app.route("/api/dataset/active", methods=["GET"])
def get_active_dataset():
    """Return currently active dataset configuration."""
    return jsonify(get_active_dataset_info())


@app.route("/api/dataset/select_source", methods=["POST"])
def select_dataset_source():
    """Switch between 'synthetic' and 'custom' active dataset."""
    data = request.get_json() or {}
    source = data.get("source", "synthetic").strip().lower()
    if source not in ["synthetic", "custom"]:
        return jsonify({"success": False, "error": "Invalid source. Choose 'synthetic' or 'custom'."}), 400

    try:
        updated = set_active_dataset_source(source=source)
        return jsonify({"success": True, "active_dataset": updated})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/dataset/analytics", methods=["GET"])
def get_dataset_analytics():
    """Compute and return rich analytics on the currently active dataset."""
    try:
        df, features, target, _ = get_active_dataframe()
        analytics = compute_dataset_analytics(df, features, target_column=target)
        return jsonify(analytics)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to compute dataset analytics: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# NEW: Model Training Studio Routes (Async Background Execution)
# ---------------------------------------------------------------------------

def _run_training_worker(params):
    """Background worker for model training to avoid browser timeouts."""
    global training_state

    with training_state_lock:
        training_state["status"] = "running"
        training_state["progress"] = 5
        training_state["stage"] = "Initializing training pipeline..."
        training_state["logs"] = ["[Info] Training pipeline initialized."]
        training_state["error"] = None
        training_state["results"] = None
        training_state["cancel_requested"] = False
        training_state["start_time"] = time.time()

    def add_log(msg):
        with training_state_lock:
            training_state["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            print(f"[Training] {msg}")

    try:
        # Load active dataset
        add_log("Loading active dataset...")
        df, features, target, class_labels = get_active_dataframe()
        
        with training_state_lock:
            training_state["progress"] = 15
            training_state["stage"] = "Splitting and normalizing dataset..."

        test_size = float(params.get("test_size", 0.25))
        random_seed = int(params.get("random_seed", RANDOM_SEED))
        model_choice = params.get("model_choice", "classical").lower()
        q_samples = int(params.get("quantum_samples", 80))

        add_log(f"Active dataset: {len(df)} samples, {len(features)} features, target: '{target}'.")
        add_log(f"Configuring train/test split (test_size={test_size}, random_seed={random_seed})...")

        # Partition data with zero leakage
        from src.preprocess import prepare_classical_data
        data = prepare_classical_data(
            df=df,
            test_size=test_size,
            random_state=random_seed,
            feature_columns=features,
            target_column=target,
        )
        X_train, X_test, y_train, y_test, scaler = data[:5]

        add_log(f"Partitioned: {len(X_train)} training samples, {len(X_test)} testing samples.")

        classical_res = None
        quantum_res = None

        # 1. Classical SVM Training
        if model_choice in ["classical", "both"]:
            with training_state_lock:
                training_state["progress"] = 30
                training_state["stage"] = "Training Classical Support Vector Machine..."

            add_log("Training Classical Support Vector Machine (RBF Kernel)...")
            classical_res = train_classical_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                scaler=scaler,
                feature_names=features,
                class_labels=class_labels,
            )
            add_log(f"Classical SVM finished in {classical_res['train_time']:.3f}s with {classical_res['accuracy']*100:.2f}% accuracy.")

        # 2. Quantum QSVC Training
        if model_choice in ["quantum", "both"]:
            with training_state_lock:
                if training_state.get("cancel_requested"):
                    add_log("Training canceled by user request.")
                    training_state["status"] = "idle"
                    return

                training_state["progress"] = 55
                training_state["stage"] = "Constructing Quantum Feature Map & Kernel..."

            add_log(f"Initiating Quantum QSVC simulation (subsampling to {q_samples} train points)...")
            q_test_samples = max(int(q_samples * test_size), 10)

            quantum_res = train_quantum_model(
                train_samples=q_samples,
                test_samples=q_test_samples,
                X_train_custom=X_train,
                y_train_custom=y_train,
                X_test_custom=X_test,
                y_test_custom=y_test,
                feature_names=features,
                class_labels=class_labels,
            )
            add_log(f"Quantum QSVC simulation finished in {quantum_res['train_time']:.2f}s with {quantum_res['accuracy']*100:.2f}% accuracy.")

        with training_state_lock:
            training_state["progress"] = 85
            training_state["stage"] = "Synthesizing evaluation metrics & diagnostic heatmaps..."

        # 3. Comparative Evaluation
        if classical_res is None:
            # Create a placeholder classical structure if only quantum was trained
            classical_res = {
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "train_time": 0.0, "predict_time": 0.0, "train_samples": 0, "test_samples": 0,
                "confusion_matrix": [], "classification_report_str": "Not evaluated",
            }

        eval_summary = evaluate_models(classical_res, quantum_res)
        add_log("Comparative metrics and confusion matrices saved to results/.")

        # 4. Log to Experiment History
        active_meta = get_active_dataset_info()
        dataset_name = active_meta.get("dataset_name", "Synthetic Telemetry")
        dataset_src = active_meta.get("source", "synthetic")

        if model_choice == "both" and quantum_res:
            log_experiment({
                "dataset_name": dataset_name,
                "dataset_source": dataset_src,
                "model_type": "Hybrid Benchmark (Classical & Quantum)",
                "features_count": len(features),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracy": classical_res.get("accuracy", 0.0),
                "precision": classical_res.get("precision", 0.0),
                "recall": classical_res.get("recall", 0.0),
                "macro_f1": classical_res.get("f1", 0.0),
                "weighted_f1": classical_res.get("weighted_f1", classical_res.get("f1", 0.0)),
                "train_time_sec": classical_res.get("train_time", 0.0),
                "predict_time_sec": classical_res.get("predict_time", 0.0),
                "status": "SUCCESS",
                "notes": f"Classical: {classical_res['accuracy']*100:.1f}%, Quantum: {quantum_res['accuracy']*100:.1f}%",
            })
        elif classical_res and model_choice == "classical":
            log_experiment({
                "dataset_name": dataset_name,
                "dataset_source": dataset_src,
                "model_type": "Classical SVM (RBF)",
                "features_count": len(features),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "accuracy": classical_res.get("accuracy", 0.0),
                "precision": classical_res.get("precision", 0.0),
                "recall": classical_res.get("recall", 0.0),
                "macro_f1": classical_res.get("f1", 0.0),
                "weighted_f1": classical_res.get("weighted_f1", classical_res.get("f1", 0.0)),
                "train_time_sec": classical_res.get("train_time", 0.0),
                "predict_time_sec": classical_res.get("predict_time", 0.0),
                "status": "SUCCESS",
            })
        elif quantum_res:
            log_experiment({
                "dataset_name": dataset_name,
                "dataset_source": dataset_src,
                "model_type": "Quantum QSVC (ZZFeatureMap)",
                "features_count": len(features),
                "train_samples": quantum_res.get("train_samples", 0),
                "test_samples": quantum_res.get("test_samples", 0),
                "accuracy": quantum_res.get("accuracy", 0.0),
                "precision": quantum_res.get("precision", 0.0),
                "recall": quantum_res.get("recall", 0.0),
                "macro_f1": quantum_res.get("f1", 0.0),
                "weighted_f1": quantum_res.get("weighted_f1", quantum_res.get("f1", 0.0)),
                "train_time_sec": quantum_res.get("train_time", 0.0),
                "predict_time_sec": quantum_res.get("predict_time", 0.0),
                "status": "SUCCESS",
            })

        with training_state_lock:
            training_state["progress"] = 100
            training_state["stage"] = "Training successfully completed!"
            training_state["status"] = "completed"
            training_state["results"] = eval_summary
            training_state["end_time"] = time.time()
            add_log("Training and evaluation successfully completed.")

    except Exception as e:
        with training_state_lock:
            training_state["status"] = "failed"
            training_state["error"] = str(e)
            training_state["stage"] = f"Training failed: {str(e)}"
            add_log(f"ERROR: {str(e)}")


@app.route("/api/train", methods=["POST"])
def start_training():
    """Trigger model training in background thread."""
    global training_state

    with training_state_lock:
        if training_state["status"] == "running":
            return jsonify({
                "success": False,
                "error": "A training session is already running. Please wait for it to complete."
            }), 409

    params = request.get_json() or {}
    worker = threading.Thread(target=_run_training_worker, args=(params,), daemon=True)
    worker.start()

    return jsonify({
        "success": True,
        "message": "Training session started successfully in background.",
        "status": "running"
    })


@app.route("/api/train/status", methods=["GET"])
def get_training_status():
    """Poll training progress and status."""
    with training_state_lock:
        return jsonify({
            "status": training_state["status"],
            "progress": training_state["progress"],
            "stage": training_state["stage"],
            "logs": training_state["logs"][-20:],
            "error": training_state["error"],
            "has_results": training_state["results"] is not None,
        })


@app.route("/api/train/cancel", methods=["POST"])
def cancel_training():
    """Request training cancellation."""
    global training_state
    with training_state_lock:
        if training_state["status"] == "running":
            training_state["cancel_requested"] = True
            return jsonify({"success": True, "message": "Cancellation requested."})
        return jsonify({"success": False, "message": "No active training run to cancel."})


# ---------------------------------------------------------------------------
# NEW: Batch Prediction & CSV Inference Route
# ---------------------------------------------------------------------------

@app.route("/api/predict_batch", methods=["POST"])
def predict_batch():
    """
    Perform batch fault classification on an uploaded CSV file.
    Validates compatible features against the trained model and returns
    predictions summary and first 25 samples.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"success": False, "error": "Uploaded file must be a CSV."}), 400

    model_type = request.form.get("model_type", "classical").strip().lower()

    try:
        df = pd.read_csv(file)
        if df.empty:
            return jsonify({"success": False, "error": "Uploaded CSV is empty."}), 400

        # Load active model metadata
        meta_path = os.path.join(RESULTS_DIR, "classical_metadata.joblib")
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            expected_features = meta.get("features", FEATURE_COLUMNS)
            class_labels = meta.get("class_labels", FAULT_CLASSES)
        else:
            expected_features = FEATURE_COLUMNS
            class_labels = FAULT_CLASSES

        # Verify compatible columns
        missing_cols = [c for c in expected_features if c not in df.columns]
        if len(missing_cols) > len(expected_features) / 2:
            return jsonify({
                "success": False,
                "error": f"Uploaded CSV does not match model features. Missing columns: {missing_cols[:5]}..."
            }), 400

        # Load model and scaler
        if not os.path.exists(CLASSICAL_MODEL_PATH) or not os.path.exists(CLASSICAL_SCALER_PATH):
            return jsonify({"success": False, "error": "Trained classical model not found. Please train model first."}), 400

        model = joblib.load(CLASSICAL_MODEL_PATH)
        scaler = joblib.load(CLASSICAL_SCALER_PATH)

        # Prepare matrix
        X_mat = []
        for _, row in df.iterrows():
            vec = []
            for col in expected_features:
                if col in row:
                    try:
                        vec.append(float(row[col]))
                    except (ValueError, TypeError):
                        vec.append(0.0)
                else:
                    vec.append(0.0)
            X_mat.append(vec)

        X_arr = np.array(X_mat)
        X_scaled = scaler.transform(X_arr)

        predictions = model.predict(X_scaled)
        
        # Probabilities if available
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)

        result_rows = []
        class_counts = {}

        for i, pred_cls in enumerate(predictions):
            p_cls = int(pred_cls)
            p_lbl = class_labels.get(p_cls, f"Class {p_cls}")
            class_counts[p_lbl] = class_counts.get(p_lbl, 0) + 1

            conf = float(np.max(probs[i])) if probs is not None else 0.95
            adv = get_maintenance_advisory(p_cls, p_lbl)

            result_rows.append({
                "row_id": i + 1,
                "predicted_class": p_cls,
                "predicted_label": p_lbl,
                "confidence_pct": round(conf * 100, 2),
                "severity": adv["severity"],
                "recommended_action": adv["recommended_action"],
            })

        # Save batch output to CSV for download
        output_df = df.copy()
        output_df["predicted_class"] = predictions
        output_df["predicted_label"] = [class_labels.get(int(p), f"Class {p}") for p in predictions]
        if probs is not None:
            output_df["confidence_score"] = [round(float(np.max(p)), 4) for p in probs]

        ensure_dirs()
        output_df.to_csv(BATCH_PREDICTIONS_CSV, index=False)

        return jsonify({
            "success": True,
            "total_rows": len(df),
            "class_distribution": class_counts,
            "preview": result_rows[:25],
            "download_available": True,
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Batch prediction error: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# NEW: File Downloads Routes
# ---------------------------------------------------------------------------

@app.route("/api/download/<download_type>", methods=["GET"])
def download_file(download_type):
    """Serve various system exports and reports as downloadable CSV/JSON files."""
    try:
        if download_type == "cleaned_dataset":
            if os.path.exists(CLEANED_DATA_CSV):
                return send_file(CLEANED_DATA_CSV, as_attachment=True, download_name="quantumsense_cleaned_dataset.csv")
            elif os.path.exists(SENSOR_DATA_CSV):
                return send_file(SENSOR_DATA_CSV, as_attachment=True, download_name="quantumsense_synthetic_dataset.csv")
            return jsonify({"error": "No cleaned dataset found to download."}), 404

        elif download_type == "predictions":
            if os.path.exists(PREDICTIONS_CSV):
                return send_file(PREDICTIONS_CSV, as_attachment=True, download_name="quantumsense_test_predictions.csv")
            return jsonify({"error": "No test predictions file found to download."}), 404

        elif download_type == "predictions_batch":
            if os.path.exists(BATCH_PREDICTIONS_CSV):
                return send_file(BATCH_PREDICTIONS_CSV, as_attachment=True, download_name="quantumsense_batch_predictions.csv")
            return jsonify({"error": "No batch predictions file available. Run a batch prediction first."}), 404

        elif download_type == "metrics_csv":
            if not os.path.exists(METRICS_JSON):
                return jsonify({"error": "Metrics have not been generated yet."}), 404
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            rows = []
            for m_key in ["classical", "quantum"]:
                if m_key in metrics and metrics[m_key]:
                    m_data = metrics[m_key]
                    rows.append({
                        "Model": "Classical SVM" if m_key == "classical" else "Quantum QSVC",
                        "Accuracy": m_data.get("accuracy"),
                        "Macro Precision": m_data.get("precision"),
                        "Macro Recall": m_data.get("recall"),
                        "Macro F1": m_data.get("f1"),
                        "Weighted F1": m_data.get("weighted_f1", m_data.get("f1")),
                        "Train Time (s)": m_data.get("train_time"),
                        "Predict Time (s)": m_data.get("predict_time"),
                        "Train Samples": m_data.get("train_samples"),
                        "Test Samples": m_data.get("test_samples"),
                    })
            df_m = pd.DataFrame(rows)
            csv_str = df_m.to_csv(index=False)
            return Response(csv_str, mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=quantumsense_model_metrics.csv"})

        elif download_type == "per_class_metrics":
            if not os.path.exists(METRICS_JSON):
                return jsonify({"error": "Metrics not found."}), 404
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            report_str = metrics.get("classical", {}).get("classification_report", "")
            return Response(report_str, mimetype="text/plain", headers={"Content-Disposition": "attachment;filename=quantumsense_classification_report.txt"})

        elif download_type == "confusion_matrix":
            if not os.path.exists(METRICS_JSON):
                return jsonify({"error": "Metrics not found."}), 404
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            cm = metrics.get("classical", {}).get("confusion_matrix", [])
            df_cm = pd.DataFrame(cm)
            csv_str = df_cm.to_csv(index=False, header=[f"Pred_Class_{i}" for i in range(len(cm))])
            return Response(csv_str, mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=quantumsense_confusion_matrix.csv"})

        elif download_type == "experiment_history":
            df_h = export_history_dataframe()
            csv_str = df_h.to_csv(index=False)
            return Response(csv_str, mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=quantumsense_experiment_history.csv"})

        else:
            return jsonify({"error": f"Unknown download type: {download_type}"}), 400

    except Exception as e:
        return jsonify({"error": f"Failed to generate download: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# NEW: Experiment History Routes
# ---------------------------------------------------------------------------

@app.route("/api/experiments", methods=["GET"])
def get_experiments_list():
    """Return past experiment history records."""
    limit = request.args.get("limit", default=50, type=int)
    history = get_experiments(limit=limit)
    return jsonify({
        "success": True,
        "count": len(history),
        "experiments": history
    })


@app.route("/api/experiments/clear", methods=["POST"])
def clear_experiments_history():
    """Clear past experiment records."""
    cleared = clear_experiments()
    return jsonify({"success": cleared})


# ---------------------------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[*] Starting QuantumSense Dashboard on http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
