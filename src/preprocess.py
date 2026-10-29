# URK24CS1111
# QuantumSense - Preprocessing Module
# Data loading, validation, leakage-free train/test splitting, and feature scaling

"""
preprocess.py - Preprocessing pipeline for QuantumSense.

Provides:
  - load_dataset(): Loads and validates sensor dataset from CSV
  - validate_data(): Verifies integrity, missing values, infinities, columns, class distribution
  - prepare_classical_data(): Stratified train/test split with StandardScaler fit strictly on train
  - prepare_quantum_data(): Mutual-information feature selection, stratified subsampling,
    and [0, pi] normalization for quantum feature maps (ZZFeatureMap)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Ensure project root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    CLASSICAL_TEST_SIZE,
    FAULT_CLASSES,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    NUM_CLASSES,
    QUANTUM_FEATURE_COUNT,
    QUANTUM_TEST_SAMPLES,
    QUANTUM_TRAIN_SAMPLES,
    RANDOM_SEED,
    SENSOR_DATA_CSV,
    TARGET_COLUMN,
    ensure_dirs,
)


def validate_data(df: pd.DataFrame) -> Dict[str, Union[bool, int, List[str], Dict[int, int]]]:
    """
    Validate sensor dataset for data integrity, missing/infinite values,
    required columns, and class balance.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.

    Returns
    -------
    dict
        Validation report dictionary with keys:
          - valid: bool, True if all critical checks pass
          - num_samples: int, total row count
          - num_features: int, number of recognized feature columns present
          - missing_columns: list of missing required column names
          - nan_count: int, total count of NaN/null values
          - inf_count: int, total count of +/- infinity values
          - class_distribution: dict mapping fault_class to sample count
          - issues: list of human-readable problem descriptions
    """
    issues: List[str] = []

    if df is None:
        return {
            "valid": False,
            "num_samples": 0,
            "num_features": 0,
            "missing_columns": FEATURE_COLUMNS + [TARGET_COLUMN, LABEL_COLUMN],
            "nan_count": 0,
            "inf_count": 0,
            "class_distribution": {},
            "issues": ["Input DataFrame is None."],
        }

    if not isinstance(df, pd.DataFrame):
        return {
            "valid": False,
            "num_samples": 0,
            "num_features": 0,
            "missing_columns": FEATURE_COLUMNS + [TARGET_COLUMN, LABEL_COLUMN],
            "nan_count": 0,
            "inf_count": 0,
            "class_distribution": {},
            "issues": [f"Expected pd.DataFrame, received {type(df).__name__}."],
        }

    if df.empty:
        return {
            "valid": False,
            "num_samples": 0,
            "num_features": 0,
            "missing_columns": FEATURE_COLUMNS + [TARGET_COLUMN, LABEL_COLUMN],
            "nan_count": 0,
            "inf_count": 0,
            "class_distribution": {},
            "issues": ["DataFrame is completely empty (0 rows)."],
        }

    # Check required columns
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN, LABEL_COLUMN]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Check NaNs across the entire DataFrame
    nan_count = int(df.isna().sum().sum())
    if nan_count > 0:
        issues.append(f"Found {nan_count} NaN/null values in dataset.")

    # Check Infs across numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric_cols.to_numpy()).sum()) if not numeric_cols.empty else 0
    if inf_count > 0:
        issues.append(f"Found {inf_count} infinite values in numeric columns.")

    # Check class distribution
    class_dist: Dict[int, int] = {}
    if TARGET_COLUMN in df.columns:
        class_dist = {int(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().items()}
        # Check for unexpected classes
        valid_classes = set(FAULT_CLASSES.keys())
        found_classes = set(class_dist.keys())
        unexpected = found_classes - valid_classes
        if unexpected:
            issues.append(f"Dataset contains unexpected fault class IDs: {unexpected}")
        missing_classes = valid_classes - found_classes
        if missing_classes:
            issues.append(f"Dataset is missing expected fault classes: {missing_classes}")

    is_valid = len(issues) == 0
    present_features = [c for c in FEATURE_COLUMNS if c in df.columns]

    return {
        "valid": is_valid,
        "num_samples": len(df),
        "num_features": len(present_features),
        "missing_columns": missing_cols,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "class_distribution": class_dist,
        "issues": issues,
    }


def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the sensor dataset from CSV and validate its structure and values.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. If None, uses SENSOR_DATA_CSV from config.

    Returns
    -------
    pd.DataFrame
        Validated sensor dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset CSV does not exist at the specified path.
    ValueError
        If the file is empty or fails data integrity validation.
    """
    if filepath is None:
        filepath = SENSOR_DATA_CSV

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Sensor dataset file not found at '{filepath}'. "
            f"Please run 'src/generate_data.py' to generate the dataset first."
        )

    if os.path.getsize(filepath) == 0:
        raise ValueError(f"Sensor dataset file at '{filepath}' is empty (0 bytes).")

    df = pd.read_csv(filepath)

    report = validate_data(df)
    if not report["valid"]:
        raise ValueError(
            f"Dataset validation failed for '{filepath}': {'; '.join(report['issues'])}"
        )

    return df


def prepare_classical_data(
    df: Optional[pd.DataFrame] = None,
    test_size: float = CLASSICAL_TEST_SIZE,
    random_state: int = RANDOM_SEED,
    feature_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare sensor data for classical machine learning models.

    Performs stratified train/test splitting and fits a StandardScaler
    exclusively on the training split to guarantee zero data leakage.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input DataFrame. If None, loads from SENSOR_DATA_CSV.
    test_size : float, optional
        Proportion of dataset to include in the test split (default=CLASSICAL_TEST_SIZE=0.25).
    random_state : int, optional
        Random seed for reproducibility (default=RANDOM_SEED=42).
    feature_columns : list of str, optional
        Explicit feature column names to select. Defaults to FEATURE_COLUMNS.
    target_column : str, optional
        Target class column name. Defaults to TARGET_COLUMN ('fault_class').

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test, scaler)
      - X_train: np.ndarray, standardized training feature matrix
      - X_test: np.ndarray, standardized test feature matrix
      - y_train: np.ndarray, 1D integer training labels
      - y_test: np.ndarray, 1D integer test labels
      - scaler: StandardScaler, fitted exclusively on X_train
    """
    if df is None:
        df = load_dataset()
    else:
        # If standard feature columns are present, run full validation
        has_std_cols = all(c in df.columns for c in FEATURE_COLUMNS + [TARGET_COLUMN])
        if has_std_cols and feature_columns is None:
            report = validate_data(df)
            if not report["valid"]:
                raise ValueError(f"Data validation failed: {'; '.join(report['issues'])}")

    # Resolve feature columns
    if feature_columns is not None:
        feat_cols = [c for c in feature_columns if c in df.columns]
    elif all(c in df.columns for c in FEATURE_COLUMNS):
        feat_cols = FEATURE_COLUMNS
    else:
        # Auto-detect numeric feature columns
        tgt_candidate = target_column or TARGET_COLUMN
        feat_cols = [
            c for c in df.columns
            if c != tgt_candidate and c not in ["fault_label", LABEL_COLUMN] and pd.api.types.is_numeric_dtype(df[c])
        ]

    if not feat_cols:
        raise ValueError("No valid numerical feature columns found in dataset.")

    # Resolve target column
    if target_column and target_column in df.columns:
        tgt_col = target_column
    elif TARGET_COLUMN in df.columns:
        tgt_col = TARGET_COLUMN
    elif "fault_class" in df.columns:
        tgt_col = "fault_class"
    else:
        raise ValueError("No target classification column found in dataset.")

    X_raw = df[feat_cols].to_numpy(dtype=float)
    y_raw = df[tgt_col].to_numpy(dtype=int)

    # Stratified split when sample count per class allows, otherwise standard split
    classes, counts = np.unique(y_raw, return_counts=True)
    n_cls = len(classes)
    n_test = int(np.ceil(len(X_raw) * test_size))
    n_train = len(X_raw) - n_test

    if np.min(counts) >= 2 and n_train >= n_cls and n_test >= n_cls:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y_raw,
            test_size=test_size,
            stratify=y_raw,
            random_state=random_state,
        )
    else:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw,
            y_raw,
            test_size=test_size,
            random_state=random_state,
        )

    # Strictly fit scaler on training data only to prevent data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, scaler


def prepare_quantum_data(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray, List[int]],
    y_test: Union[pd.Series, np.ndarray, List[int]],
    feature_names: Optional[List[str]] = None,
    quantum_features: int = QUANTUM_FEATURE_COUNT,
    n_train_samples: int = QUANTUM_TRAIN_SAMPLES,
    n_test_samples: int = QUANTUM_TEST_SAMPLES,
    random_state: int = RANDOM_SEED,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Pipeline]:
    if "feature_count" in kwargs:
        quantum_features = kwargs["feature_count"]
    if "train_samples" in kwargs:
        n_train_samples = kwargs["train_samples"]
    if "test_samples" in kwargs:
        n_test_samples = kwargs["test_samples"]
    """
    Prepare feature subset and subsampled partitions for quantum kernel / QSVC training.

    Steps (guaranteeing zero data leakage):
      1. Extract top `quantum_features` using mutual_info_classif strictly on training split.
      2. Subsample `n_train_samples` train and `n_test_samples` test points using stratified sampling.
      3. Fit a new StandardScaler and [0, pi] MinMaxScaler strictly on quantum training samples.
      4. Transform test samples with the fitted pipeline, clipping to [0, pi] for quantum angle encoding.

    Parameters
    ----------
    X_train : array-like of shape (n_train_samples, n_features)
        Training feature matrix (raw or pre-scaled).
    X_test : array-like of shape (n_test_samples, n_features)
        Test feature matrix.
    y_train : array-like of shape (n_train_samples,)
        Training labels.
    y_test : array-like of shape (n_test_samples,)
        Test labels.
    feature_names : list of str, optional
        Names of the input features. Inferred from DataFrame or FEATURE_COLUMNS if None.
    quantum_features : int, optional
        Number of features to select (default=QUANTUM_FEATURE_COUNT=4).
    n_train_samples : int, optional
        Target training set size for quantum model (default=QUANTUM_TRAIN_SAMPLES=80).
    n_test_samples : int, optional
        Target test set size for quantum model (default=QUANTUM_TEST_SAMPLES=40).
    random_state : int, optional
        Random seed for feature selection and subsampling reproducibility.

    Returns
    -------
    tuple of (X_train_q, X_test_q, y_train_q, y_test_q, selected_features, scaler_q)
      - X_train_q: np.ndarray of shape (n_train_samples, quantum_features) in [0, pi]
      - X_test_q: np.ndarray of shape (n_test_samples, quantum_features) in [0, pi]
      - y_train_q: np.ndarray of shape (n_train_samples,)
      - y_test_q: np.ndarray of shape (n_test_samples,)
      - selected_features: list of str, names of the selected features
      - scaler_q: Pipeline, fitted scaler mapping raw/classical features to [0, pi]
    """
    # Resolve feature names
    if isinstance(X_train, pd.DataFrame):
        names = list(X_train.columns)
        X_tr = X_train.to_numpy(dtype=float)
    else:
        X_tr = np.asarray(X_train, dtype=float)
        if feature_names is not None:
            names = list(feature_names)
        elif X_tr.shape[1] == len(FEATURE_COLUMNS):
            names = list(FEATURE_COLUMNS)
        else:
            names = [f"feature_{i}" for i in range(X_tr.shape[1])]

    if isinstance(X_test, pd.DataFrame):
        X_te = X_test.to_numpy(dtype=float)
    else:
        X_te = np.asarray(X_test, dtype=float)

    y_tr = np.asarray(y_train, dtype=int)
    y_te = np.asarray(y_test, dtype=int)

    # 1. Feature selection: mutual_info_classif strictly on training set
    mi_scores = mutual_info_classif(X_tr, y_tr, random_state=random_state)
    top_indices = np.argsort(mi_scores)[::-1][:quantum_features]
    selected_features = [names[i] for i in top_indices]

    X_tr_selected = X_tr[:, top_indices]
    X_te_selected = X_te[:, top_indices]

    # 2. Stratified subsampling for quantum training and test partitions
    def _subsample(X, y, target_n, seed):
        if target_n >= len(X):
            return X.copy(), y.copy()

        classes, counts = np.unique(y, return_counts=True)
        n_cls = len(classes)
        can_stratify = (
            np.min(counts) >= 2
            and target_n >= n_cls
            and (len(X) - target_n) >= n_cls
        )
        if can_stratify:
            try:
                X_sub, _, y_sub, _ = train_test_split(
                    X, y,
                    train_size=target_n,
                    stratify=y,
                    random_state=seed,
                )
                return X_sub, y_sub
            except ValueError:
                pass

        # Robust fallback when strict StratifiedShuffleSplit constraints cannot be satisfied
        rng = np.random.RandomState(seed)
        selected_indices = []
        indices_by_class = {c: np.where(y == c)[0] for c in classes}
        if target_n >= n_cls:
            for c in classes:
                chosen = rng.choice(indices_by_class[c], size=1, replace=False)[0]
                selected_indices.append(chosen)

        remaining_pool = [i for i in range(len(X)) if i not in selected_indices]
        slots_needed = target_n - len(selected_indices)
        if slots_needed > 0 and remaining_pool:
            chosen_remaining = rng.choice(remaining_pool, size=min(slots_needed, len(remaining_pool)), replace=False)
            selected_indices.extend(chosen_remaining.tolist())

        selected_indices = np.array(selected_indices[:target_n])
        return X[selected_indices], y[selected_indices]

    n_train = min(n_train_samples, len(X_tr_selected))
    X_tr_sub, y_train_q = _subsample(X_tr_selected, y_tr, n_train, random_state)

    n_test = min(n_test_samples, len(X_te_selected))
    X_te_sub, y_test_q = _subsample(X_te_selected, y_te, n_test, random_state + 1)

    # 3. Fit a new StandardScaler and [0, pi] MinMaxScaler strictly on quantum train subset
    scaler_q = Pipeline([
        ("standard", StandardScaler()),
        ("minmax", MinMaxScaler(feature_range=(0.0, np.pi), clip=True)),
    ])

    X_train_q = scaler_q.fit_transform(X_tr_sub)
    X_test_q = scaler_q.transform(X_te_sub)

    # Attach convenience attributes for direct access if needed
    scaler_q.standard_scaler = scaler_q.named_steps["standard"]
    scaler_q.minmax_scaler = scaler_q.named_steps["minmax"]

    return (
        np.asarray(X_train_q, dtype=float),
        np.asarray(X_test_q, dtype=float),
        np.asarray(y_train_q, dtype=int),
        np.asarray(y_test_q, dtype=int),
        selected_features,
        scaler_q,
    )


# Backward-compatibility helpers
def prepare_train_test_split(
    df: pd.DataFrame,
    test_size: float = CLASSICAL_TEST_SIZE,
    random_state: int = RANDOM_SEED,
):
    """Backward-compatible helper returning raw split."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def fit_scaler(X_train: Union[pd.DataFrame, np.ndarray]) -> StandardScaler:
    """Fit and return a StandardScaler on training data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_features(X: Union[pd.DataFrame, np.ndarray], scaler: StandardScaler) -> np.ndarray:
    """Transform feature matrix using a fitted scaler."""
    return scaler.transform(X)


if __name__ == "__main__":
    print("=" * 60)
    print("QuantumSense - Data Preprocessing Demonstration")
    print("=" * 60)

    # 1. Load dataset (generate first if missing)
    if not os.path.exists(SENSOR_DATA_CSV):
        print(f"[INFO] '{SENSOR_DATA_CSV}' not found. Generating dataset first...")
        from src.generate_data import generate_dataset
        generate_dataset()

    print(f"\n[1] Loading dataset from: {SENSOR_DATA_CSV}")
    df = load_dataset()
    print(f"    Loaded dataset shape: {df.shape}")

    # 2. Validate dataset
    print("\n[2] Validating dataset integrity...")
    report = validate_data(df)
    print(f"    Validation status:  {'PASSED' if report['valid'] else 'FAILED'}")
    print(f"    Total samples:      {report['num_samples']}")
    print(f"    Total features:     {report['num_features']}")
    print(f"    Missing columns:    {report['missing_columns']}")
    print(f"    NaN count:          {report['nan_count']}")
    print(f"    Inf count:          {report['inf_count']}")
    print(f"    Class distribution: {report['class_distribution']}")

    # 3. Classical data preparation
    print("\n[3] Preparing classical train/test splits...")
    X_train_c, X_test_c, y_train_c, y_test_c, scaler_c = prepare_classical_data(df)
    print(f"    Classical X_train shape: {X_train_c.shape}")
    print(f"    Classical X_test shape:  {X_test_c.shape}")
    print(f"    Classical y_train shape: {y_train_c.shape}")
    print(f"    Classical y_test shape:  {y_test_c.shape}")
    print(f"    Scaler mean: min={scaler_c.mean_.min():.2f}, max={scaler_c.mean_.max():.2f}")

    # 4. Quantum data preparation
    print("\n[4] Preparing quantum feature subset and subsampled data...")
    X_train_q, X_test_q, y_train_q, y_test_q, sel_feats, scaler_q = prepare_quantum_data(
        X_train_c, X_test_c, y_train_c, y_test_c
    )
    print(f"    Selected quantum features ({len(sel_feats)}): {sel_feats}")
    print(f"    Quantum X_train shape:   {X_train_q.shape}")
    print(f"    Quantum X_test shape:    {X_test_q.shape}")
    print(f"    Quantum y_train shape:   {y_train_q.shape}")
    print(f"    Quantum y_test shape:    {y_test_q.shape}")
    print(f"    Quantum angle range X_train: [{X_train_q.min():.4f}, {X_train_q.max():.4f}] rad (expected [0, pi])")
    print(f"    Quantum angle range X_test:  [{X_test_q.min():.4f}, {X_test_q.max():.4f}] rad")

    print("\n[SUCCESS] Preprocessing demonstration completed successfully.")
