# URK24CS1111
# QuantumSense - Experiment History Tracker Module
# Records and manages model training experiments, configurations, and evaluation metrics.

"""
experiment_tracker.py - Manages experiment logging, persistence, retrieval,
and CSV export for all classical and quantum training runs.
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Any

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import EXPERIMENT_HISTORY_JSON, ensure_dirs


def log_experiment(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log a completed training experiment to EXPERIMENT_HISTORY_JSON.

    Args:
        data (dict): Experiment details including dataset name, models, metrics.

    Returns:
        dict: The recorded experiment entry with generated ID and timestamp.
    """
    ensure_dirs()

    exp_id = f"EXP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "experiment_id": exp_id,
        "timestamp": timestamp,
        "dataset_name": data.get("dataset_name", "Synthetic Telemetry"),
        "dataset_source": data.get("dataset_source", "synthetic"),
        "model_type": data.get("model_type", "Classical SVM"),
        "features_count": data.get("features_count", 15),
        "train_samples": data.get("train_samples", 0),
        "test_samples": data.get("test_samples", 0),
        "accuracy": round(float(data.get("accuracy", 0.0)), 4),
        "precision": round(float(data.get("precision", 0.0)), 4),
        "recall": round(float(data.get("recall", 0.0)), 4),
        "macro_f1": round(float(data.get("macro_f1", 0.0)), 4),
        "weighted_f1": round(float(data.get("weighted_f1", 0.0)), 4),
        "train_time_sec": round(float(data.get("train_time_sec", 0.0)), 4),
        "predict_time_sec": round(float(data.get("predict_time_sec", 0.0)), 4),
        "status": data.get("status", "SUCCESS"),
        "notes": data.get("notes", ""),
    }

    history = []
    if os.path.exists(EXPERIMENT_HISTORY_JSON):
        try:
            with open(EXPERIMENT_HISTORY_JSON, "r", encoding="utf-8") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except Exception:
            history = []

    history.insert(0, entry)  # Prepend newest

    # Limit to latest 100 experiments
    history = history[:100]

    with open(EXPERIMENT_HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return entry


def get_experiments(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve past training experiment history.

    Args:
        limit (int): Maximum number of experiments to return.

    Returns:
        list: List of experiment dictionaries ordered newest first.
    """
    if not os.path.exists(EXPERIMENT_HISTORY_JSON):
        return []

    try:
        with open(EXPERIMENT_HISTORY_JSON, "r", encoding="utf-8") as f:
            history = json.load(f)
            if isinstance(history, list):
                return history[:limit]
            return []
    except Exception:
        return []


def clear_experiments() -> bool:
    """Clear all recorded experiments from history."""
    ensure_dirs()
    try:
        with open(EXPERIMENT_HISTORY_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return True
    except Exception:
        return False


def export_history_dataframe() -> pd.DataFrame:
    """
    Export experiment history as a clean Pandas DataFrame for CSV download.

    Returns:
        pd.DataFrame: Tabular experiment history.
    """
    history = get_experiments(limit=100)
    if not history:
        return pd.DataFrame(columns=[
            "experiment_id", "timestamp", "dataset_name", "dataset_source",
            "model_type", "features_count", "train_samples", "test_samples",
            "accuracy", "precision", "recall", "macro_f1", "weighted_f1",
            "train_time_sec", "predict_time_sec", "status"
        ])
    return pd.DataFrame(history)
