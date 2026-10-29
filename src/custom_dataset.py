# URK24CS1111
# QuantumSense - Custom Dataset Lab & Analytics Module
# Handles custom CSV uploads, data inspection, sliding-window time-series extraction,
# zero-leakage preprocessing, correlation analysis, and dataset analytics.

"""
custom_dataset.py - Module for managing custom uploaded datasets,
preview inspection, Mode A (precomputed features) and Mode B (raw time-series
sliding window), data cleaning, label encoding, and dataset analytics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    DATA_DIR,
    UPLOADS_DIR,
    RESULTS_DIR,
    SENSOR_DATA_CSV,
    CUSTOM_DATA_CSV,
    ACTIVE_DATASET_CONFIG,
    CLEANED_DATA_CSV,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LABEL_COLUMN,
    FAULT_CLASSES,
    ensure_dirs,
)


def inspect_csv(filepath: str, max_preview_rows: int = 15) -> Dict[str, Any]:
    """
    Inspect an uploaded CSV file, analyze columns, datatypes,
    missing values, duplicate rows, and generate auto-suggestions.

    Args:
        filepath (str): Path to the uploaded CSV file.
        max_preview_rows (int): Number of preview rows to return.

    Returns:
        dict: Inspection report containing schema, statistics, and preview.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    file_size_bytes = os.path.getsize(filepath)
    if file_size_bytes == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    # Read CSV with encoding fallbacks
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding="latin-1")
        except Exception as e:
            raise ValueError(f"Could not parse CSV file encoding: {str(e)}")

    if df.empty or len(df.columns) == 0:
        raise ValueError("CSV contains no tabular data or columns.")

    total_rows = int(len(df))
    total_cols = int(len(df.columns))

    columns_info = []
    numeric_cols = []
    categorical_cols = []
    potential_target_cols = []
    potential_timestamp_cols = []

    for col in df.columns:
        col_str = str(col).strip()
        series = df[col]
        dtype_str = str(series.dtype)
        is_num = pd.api.types.is_numeric_dtype(series)
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique())

        col_type = "numeric" if is_num else "categorical"
        if is_num:
            numeric_cols.append(col_str)
        else:
            categorical_cols.append(col_str)

        # Check for timestamp
        lower_name = col_str.lower()
        if any(term in lower_name for term in ["time", "date", "stamp", "timestamp", "datetime"]):
            potential_timestamp_cols.append(col_str)
            col_type = "timestamp"

        # Check for potential target / fault label
        if any(term in lower_name for term in ["fault", "label", "class", "target", "status", "anomaly", "mode"]):
            potential_target_cols.insert(0, col_str)
        elif 2 <= unique_count <= 15 and col_str in categorical_cols:
            potential_target_cols.append(col_str)

        columns_info.append({
            "name": col_str,
            "type": col_type,
            "dtype": dtype_str,
            "null_count": null_count,
            "unique_count": unique_count,
            "sample_values": [str(v) for v in series.dropna().iloc[:3].tolist()]
        })

    # Pick auto-suggested selections
    suggested_target = potential_target_cols[0] if potential_target_cols else None
    suggested_timestamp = potential_timestamp_cols[0] if potential_timestamp_cols else None

    suggested_features = [
        c for c in numeric_cols
        if c != suggested_target and c != suggested_timestamp
    ]

    # Check duplicates
    duplicate_count = int(df.duplicated().sum())

    # Calculate class distribution if target is identified
    class_distribution = {}
    if suggested_target and suggested_target in df.columns:
        val_counts = df[suggested_target].value_counts().to_dict()
        class_distribution = {str(k): int(v) for k, v in val_counts.items()}

    # Preview rows safe for JSON serialization
    preview_df = df.head(max_preview_rows).copy()
    preview_rows = preview_df.replace({np.nan: None}).to_dict(orient="records")

    return {
        "success": True,
        "filename": os.path.basename(filepath),
        "file_size_kb": round(file_size_bytes / 1024, 2),
        "total_rows": total_rows,
        "total_columns": total_cols,
        "columns": columns_info,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "duplicate_rows": duplicate_count,
        "missing_values_total": int(df.isna().sum().sum()),
        "has_labels": bool(suggested_target is not None),
        "suggested_target": suggested_target,
        "suggested_timestamp": suggested_timestamp,
        "suggested_features": suggested_features,
        "class_distribution": class_distribution,
        "preview_rows": preview_rows,
    }


def extract_window_features(
    df: pd.DataFrame,
    sensor_columns: List[str],
    window_size: int = 50,
    stride: Optional[int] = None,
    target_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Mode B: Extract statistical sliding-window features from sequential time-series sensor data.
    For each window of size W and each sensor column, calculates:
      - mean
      - std (standard deviation)
      - min (minimum value)
      - max (maximum value)
      - slope (linear regression trend coefficient)

    Args:
        df (pd.DataFrame): Raw time series dataframe.
        sensor_columns (list): Columns representing sensor channels.
        window_size (int): Readings per sliding window (default 50).
        stride (int, optional): Step size between windows. Default is non-overlapping (stride = window_size).
        target_col (str, optional): Target fault label column.
        timestamp_col (str, optional): Optional timestamp column used for sorting.

    Returns:
        pd.DataFrame: Windowed features with consistent column naming.
    """
    if df is None or len(df) == 0:
        raise ValueError("Cannot extract features from empty dataframe.")

    if not sensor_columns:
        raise ValueError("At least one sensor feature column must be selected.")

    if window_size < 3:
        raise ValueError("Window size must be at least 3 readings.")

    if stride is None or stride <= 0:
        stride = window_size  # Non-overlapping windows prevent data leakage

    # Sort by timestamp if specified
    work_df = df.copy()
    if timestamp_col and timestamp_col in work_df.columns:
        try:
            work_df = work_df.sort_values(by=timestamp_col).reset_index(drop=True)
        except Exception:
            pass

    total_len = len(work_df)
    if total_len < window_size:
        raise ValueError(
            f"Dataset length ({total_len} rows) is smaller than window size ({window_size}). "
            f"Please decrease window size or upload more data."
        )

    t_axis = np.arange(window_size)
    window_records = []

    for start_idx in range(0, total_len - window_size + 1, stride):
        window = work_df.iloc[start_idx : start_idx + window_size]
        rec = {}

        # Statistical features for each sensor column
        for col in sensor_columns:
            series = pd.to_numeric(window[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            col_clean = str(col).replace(" ", "_").lower()

            c_mean = float(np.mean(series))
            c_std = float(np.std(series))
            c_min = float(np.min(series))
            c_max = float(np.max(series))

            if window_size > 1:
                slope = float(np.polyfit(t_axis, series, 1)[0])
            else:
                slope = 0.0

            rec[f"{col_clean}_mean"] = round(c_mean, 4)
            rec[f"{col_clean}_std"] = round(c_std, 4)
            rec[f"{col_clean}_min"] = round(c_min, 4)
            rec[f"{col_clean}_max"] = round(c_max, 4)
            rec[f"{col_clean}_slope"] = round(slope, 4)

        # Assign label for window if target column is provided
        if target_col and target_col in window.columns:
            labels = window[target_col].dropna()
            if len(labels) > 0:
                # Majority vote within the window
                assigned_label = labels.mode().iloc[0]
            else:
                assigned_label = "Unknown"
            rec["fault_label"] = str(assigned_label)

        window_records.append(rec)

    return pd.DataFrame(window_records)


def clean_and_preprocess(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: Optional[str] = None,
    missing_strategy: str = "mean",
    drop_duplicates: bool = False,
    handle_outliers: bool = False,
) -> Tuple[pd.DataFrame, Dict[int, str], Dict[str, Any]]:
    """
    Clean dataset and prepare features and target labels.

    Args:
        df (pd.DataFrame): Dataframe to clean.
        feature_columns (list): Numerical feature columns.
        target_column (str, optional): Target class label column.
        missing_strategy (str): 'drop', 'mean', 'median', or 'ffill'.
        drop_duplicates (bool): Whether to drop duplicate rows.
        handle_outliers (bool): Whether to clip extreme IQR outliers.

    Returns:
        tuple: (cleaned_df, label_mapping, cleaning_report)
    """
    if df is None or len(df) == 0:
        raise ValueError("Cannot clean empty dataframe.")

    initial_rows = len(df)
    clean_df = df.copy()
    report_actions = []

    # 1. Drop duplicates
    if drop_duplicates:
        dups = int(clean_df.duplicated().sum())
        if dups > 0:
            clean_df = clean_df.drop_duplicates().reset_index(drop=True)
            report_actions.append(f"Removed {dups} duplicate rows.")

    # 2. Ensure numeric features
    for col in feature_columns:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # 3. Missing values strategy
    missing_before = int(clean_df[feature_columns].isna().sum().sum())
    if missing_before > 0:
        if missing_strategy == "drop":
            clean_df = clean_df.dropna(subset=feature_columns).reset_index(drop=True)
            report_actions.append(f"Dropped rows with missing values (before: {initial_rows}, after: {len(clean_df)}).")
        elif missing_strategy == "median":
            for col in feature_columns:
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())
            report_actions.append(f"Imputed {missing_before} missing feature values with column medians.")
        elif missing_strategy == "ffill":
            clean_df[feature_columns] = clean_df[feature_columns].ffill().bfill()
            report_actions.append(f"Applied forward/backward fill for {missing_before} missing feature values.")
        else:  # mean (default)
            for col in feature_columns:
                clean_df[col] = clean_df[col].fillna(clean_df[col].mean())
            report_actions.append(f"Imputed {missing_before} missing feature values with column means.")

    # 4. Outlier handling (IQR clipping)
    if handle_outliers:
        clipped_count = 0
        for col in feature_columns:
            q25 = clean_df[col].quantile(0.25)
            q75 = clean_df[col].quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower = q25 - 2.5 * iqr
                upper = q75 + 2.5 * iqr
                clipped = ((clean_df[col] < lower) | (clean_df[col] > upper)).sum()
                clipped_count += int(clipped)
                clean_df[col] = clean_df[col].clip(lower=lower, upper=upper)
        if clipped_count > 0:
            report_actions.append(f"Clipped {clipped_count} extreme outlier values to IQR boundaries.")

    # 5. Target Label Encoding
    label_mapping = {}
    if target_column and target_column in clean_df.columns:
        clean_df = clean_df.dropna(subset=[target_column]).reset_index(drop=True)
        raw_labels = clean_df[target_column].astype(str)
        unique_labels = sorted(raw_labels.unique())

        label_to_id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        label_mapping = {idx: lbl for idx, lbl in enumerate(unique_labels)}

        clean_df["fault_class"] = raw_labels.map(label_to_id).astype(int)
        clean_df["fault_label"] = raw_labels
        report_actions.append(
            f"Encoded {len(unique_labels)} target classes into integer IDs (0..{len(unique_labels)-1}): {label_mapping}"
        )
    else:
        report_actions.append("No target column provided; dataset configured for inference only.")

    cleaning_report = {
        "initial_rows": initial_rows,
        "final_rows": len(clean_df),
        "feature_count": len(feature_columns),
        "actions": report_actions,
        "has_labels": bool(target_column and target_column in clean_df.columns),
        "classes": label_mapping,
    }

    return clean_df, label_mapping, cleaning_report


def compute_dataset_analytics(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: Optional[str] = None,
    max_corr_features: int = 12,
) -> Dict[str, Any]:
    """
    Compute comprehensive dataset analytics:
      - Memory & dimension metrics
      - Class distribution & class imbalance warning
      - Feature summary statistics (mean, std, min, max, skew)
      - Correlation matrix for numerical features
      - Distribution histogram data for visualization

    Args:
        df (pd.DataFrame): Dataset to analyze.
        feature_columns (list): List of feature column names.
        target_column (str, optional): Target column name.
        max_corr_features (int): Maximum number of features in correlation matrix.

    Returns:
        dict: Analytics payload ready for Chart.js and UI components.
    """
    if df is None or len(df) == 0:
        return {"error": "Empty dataset provided."}

    total_rows = int(len(df))
    valid_features = [c for c in feature_columns if c in df.columns]

    # Memory usage in MB
    memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 2)
    missing_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    # Class distribution & imbalance
    class_dist = {}
    imbalance_warning = None
    if target_column and target_column in df.columns:
        counts = df[target_column].value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in counts.items()}
        if len(counts) >= 2:
            min_c = min(counts.values())
            max_c = max(counts.values())
            ratio = min_c / max_c
            if ratio < 0.25:
                imbalance_warning = (
                    f"Severe class imbalance detected: minority class has {min_c} samples vs majority {max_c} "
                    f"(ratio: {ratio:.2f}). Consider stratified splitting or class reweighting."
                )

    # Summary statistics for features
    stats = []
    for col in valid_features:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) > 0:
            stats.append({
                "feature": col,
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "p25": round(float(series.quantile(0.25)), 4),
                "median": round(float(series.median()), 4),
                "p75": round(float(series.quantile(0.75)), 4),
                "max": round(float(series.max()), 4),
                "skew": round(float(series.skew()), 4) if len(series) > 2 else 0.0,
            })

    # Correlation matrix
    corr_cols = valid_features[:max_corr_features]
    corr_data = []
    if len(corr_cols) > 1:
        corr_matrix = df[corr_cols].corr().fillna(0.0)
        corr_matrix_values = [[round(float(corr_matrix.iloc[i, j]), 2) for j in range(len(corr_cols))] for i in range(len(corr_cols))]
        corr_data = {
            "columns": corr_cols,
            "matrix": corr_matrix_values,
        }

    # Distribution histogram bins for top 4 features
    histograms = {}
    for col in valid_features[:4]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) > 1:
            counts, bin_edges = np.histogram(series, bins=10)
            histograms[col] = {
                "labels": [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(counts))],
                "counts": [int(c) for c in counts],
            }

    return {
        "success": True,
        "total_rows": total_rows,
        "feature_count": len(valid_features),
        "total_columns": int(len(df.columns)),
        "memory_mb": memory_mb,
        "missing_count": missing_count,
        "duplicate_count": duplicate_count,
        "class_distribution": class_dist,
        "imbalance_warning": imbalance_warning,
        "feature_statistics": stats,
        "correlation": corr_data,
        "histograms": histograms,
    }


def get_active_dataset_info() -> Dict[str, Any]:
    """
    Get metadata regarding the currently active dataset (Synthetic or Custom).

    Returns:
        dict: Active dataset properties, path, source, and column info.
    """
    ensure_dirs()
    if os.path.exists(ACTIVE_DATASET_CONFIG):
        try:
            with open(ACTIVE_DATASET_CONFIG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                source = cfg.get("source", "synthetic")
                if source == "custom" and os.path.exists(CUSTOM_DATA_CSV):
                    return cfg
        except Exception:
            pass

    # Default: Synthetic Dataset
    synthetic_exists = os.path.exists(SENSOR_DATA_CSV)
    row_count = 0
    if synthetic_exists:
        try:
            row_count = len(pd.read_csv(SENSOR_DATA_CSV))
        except Exception:
            row_count = 1000

    return {
        "source": "synthetic",
        "dataset_name": "Synthetic Multi-Sensor Telemetry",
        "filepath": SENSOR_DATA_CSV,
        "total_rows": row_count,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "label_column": LABEL_COLUMN,
        "fault_classes": FAULT_CLASSES,
        "is_custom": False,
    }


def set_active_dataset_source(source: str = "synthetic", custom_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Switch active dataset source between 'synthetic' and 'custom'.
    Preserves both datasets on disk without deletion.

    Args:
        source (str): 'synthetic' or 'custom'.
        custom_meta (dict, optional): Custom metadata if source is 'custom'.

    Returns:
        dict: Updated active dataset status.
    """
    ensure_dirs()
    if source == "custom":
        if not os.path.exists(CUSTOM_DATA_CSV):
            raise FileNotFoundError("Custom dataset has not been processed and saved yet.")

        cfg = custom_meta or {}
        cfg["source"] = "custom"
        cfg["filepath"] = CUSTOM_DATA_CSV
        cfg["is_custom"] = True
    else:
        cfg = {
            "source": "synthetic",
            "dataset_name": "Synthetic Multi-Sensor Telemetry",
            "filepath": SENSOR_DATA_CSV,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "label_column": LABEL_COLUMN,
            "fault_classes": FAULT_CLASSES,
            "is_custom": False,
        }

    with open(ACTIVE_DATASET_CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return cfg


def save_custom_dataset(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """
    Save processed custom dataset to CUSTOM_DATA_CSV and update ACTIVE_DATASET_CONFIG.

    Args:
        df (pd.DataFrame): Cleaned and processed custom dataset.
        metadata (dict): Configuration metadata.

    Returns:
        str: Path where custom dataset was saved.
    """
    ensure_dirs()
    df.to_csv(CUSTOM_DATA_CSV, index=False)
    df.to_csv(CLEANED_DATA_CSV, index=False)

    metadata["source"] = "custom"
    metadata["filepath"] = CUSTOM_DATA_CSV
    metadata["total_rows"] = len(df)
    metadata["is_custom"] = True

    with open(ACTIVE_DATASET_CONFIG, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return CUSTOM_DATA_CSV


def get_active_dataframe() -> Tuple[pd.DataFrame, List[str], str, Dict[int, str]]:
    """
    Load the currently active dataset (synthetic or custom) with its feature list,
    target column, and class label mapping.

    Returns:
        tuple: (df, feature_columns, target_column, class_labels_dict)
    """
    active_info = get_active_dataset_info()
    path = active_info.get("filepath", SENSOR_DATA_CSV)

    if not os.path.exists(path):
        if path == SENSOR_DATA_CSV:
            from src.generate_data import generate_dataset
            df = generate_dataset()
        else:
            raise FileNotFoundError(f"Active dataset file not found at {path}")
    else:
        df = pd.read_csv(path)

    features = active_info.get("feature_columns", FEATURE_COLUMNS)
    target = active_info.get("target_column", TARGET_COLUMN)
    fault_classes = active_info.get("fault_classes", FAULT_CLASSES)

    # Convert keys to int if json serialized as str
    if fault_classes and isinstance(fault_classes, dict):
        fault_classes = {int(k): str(v) for k, v in fault_classes.items()}

    return df, features, target, fault_classes
