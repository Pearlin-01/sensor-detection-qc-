# URK24CS1111
# QuantumSense - Tests for Data Generation & Preprocessing Pipeline

"""
test_data.py - Pytest test suite for synthetic sensor data generation and preprocessing.
Validates dataset dimensions, column schemas, multi-class distribution, lack of NaN/Inf,
reproducibility across random seeds, data leakage prevention, and quantum preprocessing.
"""

import os
import sys
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
    CLASS_LABELS,
    SAMPLES_PER_CLASS,
    WINDOW_SIZE,
    TARGET_COLUMN,
    LABEL_COLUMN,
    SENSOR_DATA_CSV,
)
from src.generate_data import (
    generate_dataset,
    generate_sensor_dataset,
    generate_single_sample,
)
from src.preprocess import (
    load_dataset,
    validate_data,
    prepare_classical_data,
    prepare_quantum_data,
)


@pytest.fixture
def fast_generated_dataset() -> pd.DataFrame:
    """Generate a quick in-memory dataset with 10 samples per class (50 total)."""
    return generate_dataset(
        samples_per_class=10,
        random_seed=42,
        save_csv=False,
        save_plot=False,
    )


def test_generate_dataset_creates_file(tmp_path):
    """Verify generate_dataset successfully writes a CSV file to the target path."""
    csv_target = tmp_path / "test_sensor_data.csv"
    df = generate_dataset(
        samples_per_class=5,
        save_csv=True,
        save_plot=False,
        output_path=str(csv_target),
    )
    assert os.path.exists(csv_target), "CSV file was not created on disk."
    assert os.path.getsize(csv_target) > 0, "Generated CSV file is empty."
    assert isinstance(df, pd.DataFrame), "Returned object is not a DataFrame."


@pytest.mark.slow
def test_dataset_shape():
    """Verify dataset dimensions equal (SAMPLES_PER_CLASS * NUM_CLASSES, 15 features + 2 target/label)."""
    df = generate_dataset(
        samples_per_class=SAMPLES_PER_CLASS,
        save_csv=False,
        save_plot=False,
    )
    expected_rows = SAMPLES_PER_CLASS * NUM_CLASSES
    expected_cols = len(FEATURE_COLUMNS) + 2  # 15 features + fault_class + fault_label
    assert df.shape == (expected_rows, expected_cols), (
        f"Expected shape ({expected_rows}, {expected_cols}), got {df.shape}."
    )


def test_dataset_columns(fast_generated_dataset):
    """Verify all 15 feature columns, TARGET_COLUMN, and LABEL_COLUMN exist in dataset."""
    df = fast_generated_dataset
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN, LABEL_COLUMN]
    for col in expected_columns:
        assert col in df.columns, f"Required column '{col}' is missing from dataset."
    assert list(df.columns) == expected_columns, "Column ordering does not match expected schema."


def test_all_classes_present(fast_generated_dataset):
    """Verify all 5 fault classes (0 through 4) are present in the dataset."""
    df = fast_generated_dataset
    classes_found = sorted(df[TARGET_COLUMN].unique().tolist())
    expected_classes = sorted(list(FAULT_CLASSES.keys()))
    assert classes_found == expected_classes, (
        f"Expected classes {expected_classes}, but found {classes_found}."
    )

    labels_found = sorted(df[LABEL_COLUMN].unique().tolist())
    expected_labels = sorted(list(FAULT_CLASSES.values()))
    assert labels_found == expected_labels, (
        f"Expected labels {expected_labels}, but found {labels_found}."
    )


def test_class_distribution():
    """Verify that each class contains exactly the requested number of samples."""
    samples = 8
    df = generate_dataset(
        samples_per_class=samples,
        save_csv=False,
        save_plot=False,
    )
    counts = df[TARGET_COLUMN].value_counts().to_dict()
    for cls_idx in FAULT_CLASSES.keys():
        assert counts.get(cls_idx, 0) == samples, (
            f"Class {cls_idx} expected {samples} samples, got {counts.get(cls_idx, 0)}."
        )


def test_no_missing_values(fast_generated_dataset):
    """Verify there are zero NaN or missing values across all features and target columns."""
    df = fast_generated_dataset
    nan_counts = df.isna().sum()
    total_nans = int(nan_counts.sum())
    assert total_nans == 0, f"Found {total_nans} NaN values: {nan_counts[nan_counts > 0].to_dict()}."


def test_no_infinite_values(fast_generated_dataset):
    """Verify there are zero infinite values in any of the numeric feature columns."""
    df = fast_generated_dataset
    features_matrix = df[FEATURE_COLUMNS].to_numpy()
    inf_count = int(np.isinf(features_matrix).sum())
    assert inf_count == 0, f"Found {inf_count} infinite values in features."


def test_reproducibility():
    """Verify that generating twice with the identical random seed produces identical DataFrames."""
    df1 = generate_dataset(
        samples_per_class=5,
        random_seed=123,
        save_csv=False,
        save_plot=False,
    )
    df2 = generate_dataset(
        samples_per_class=5,
        random_seed=123,
        save_csv=False,
        save_plot=False,
    )
    pd.testing.assert_frame_equal(df1, df2, check_exact=True)


def test_feature_ranges(fast_generated_dataset):
    """Verify features have non-zero variance, reasonable bounds, and min <= mean <= max."""
    df = fast_generated_dataset
    for col in FEATURE_COLUMNS:
        std_val = df[col].std()
        assert std_val > 0.0, f"Feature '{col}' has zero variance (constant values)."

    # Verify physical consistency: min <= mean <= max
    assert (df["temp_min"] <= df["temp_mean"]).all()
    assert (df["temp_mean"] <= df["temp_max"]).all()

    assert (df["pres_min"] <= df["pres_mean"]).all()
    assert (df["pres_mean"] <= df["pres_max"]).all()

    assert (df["vib_min"] <= df["vib_mean"]).all()
    assert (df["vib_mean"] <= df["vib_max"]).all()


def test_generate_single_sample():
    """Verify generate_single_sample returns expected schema and keys for live inference."""
    sample = generate_single_sample(fault_class=0, seed=42)
    assert isinstance(sample, dict), "Single sample must be returned as a dictionary."

    for col in FEATURE_COLUMNS:
        assert col in sample, f"Feature column '{col}' missing from single sample."
        assert isinstance(sample[col], (int, float, np.floating)), f"Feature '{col}' is not numeric."

    assert sample[TARGET_COLUMN] == 0
    assert sample[LABEL_COLUMN] == FAULT_CLASSES[0]

    # Verify invalid class raises ValueError
    with pytest.raises(ValueError):
        generate_single_sample(fault_class=99)


def test_validate_data(fast_generated_dataset):
    """Test validation report function on valid, empty, and corrupted dataframes."""
    # 1. Valid DataFrame
    report = validate_data(fast_generated_dataset)
    assert report["valid"] is True
    assert report["nan_count"] == 0
    assert report["inf_count"] == 0
    assert len(report["missing_columns"]) == 0

    # 2. DataFrame with NaNs
    corrupted_df = fast_generated_dataset.copy()
    corrupted_df.loc[0, "temp_mean"] = np.nan
    report_nan = validate_data(corrupted_df)
    assert report_nan["valid"] is False
    assert report_nan["nan_count"] > 0

    # 3. Empty DataFrame
    empty_df = pd.DataFrame()
    report_empty = validate_data(empty_df)
    assert report_empty["valid"] is False

    # 4. Missing required columns
    dropped_df = fast_generated_dataset.drop(columns=["temp_mean", TARGET_COLUMN])
    report_dropped = validate_data(dropped_df)
    assert report_dropped["valid"] is False
    assert "temp_mean" in report_dropped["missing_columns"]
    assert TARGET_COLUMN in report_dropped["missing_columns"]


def test_load_dataset(tmp_path, sample_dataframe):
    """Verify load_dataset reads CSV file and raises FileNotFoundError on missing files."""
    csv_file = tmp_path / "valid_sensor_data.csv"
    sample_dataframe.to_csv(csv_file, index=False)

    df_loaded = load_dataset(filepath=str(csv_file))
    assert isinstance(df_loaded, pd.DataFrame)
    assert len(df_loaded) == len(sample_dataframe)
    assert list(df_loaded.columns) == list(sample_dataframe.columns)

    # Missing file test
    missing_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset(filepath=str(missing_file))


def test_prepare_classical_data(sample_dataframe):
    """Verify prepare_classical_data splits, scales, and returns scaler fit on train."""
    test_ratio = 0.33
    X_train, X_test, y_train, y_test, scaler = prepare_classical_data(
        df=sample_dataframe,
        test_size=test_ratio,
        random_state=42,
    )

    total_samples = len(sample_dataframe)
    assert X_train.shape[1] == len(FEATURE_COLUMNS)
    assert X_test.shape[1] == len(FEATURE_COLUMNS)
    assert len(X_train) + len(X_test) == total_samples
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    # Scaler was fit on train: X_train mean is close to 0 and std close to 1
    np.testing.assert_allclose(X_train.mean(axis=0), 0.0, atol=1e-1)
    np.testing.assert_allclose(X_train.std(axis=0), 1.0, atol=1e-1)


def test_prepare_quantum_data(sample_dataframe):
    """Verify quantum data preparation selects features and bounds values to [0, pi]."""
    X_train_c, X_test_c, y_train_c, y_test_c, _ = prepare_classical_data(
        df=sample_dataframe,
        test_size=0.33,
        random_state=42,
    )

    q_features = 4
    q_train_n = 6
    q_test_n = 3

    X_train_q, X_test_q, y_train_q, y_test_q, selected_features, scaler_q = prepare_quantum_data(
        X_train=X_train_c,
        X_test=X_test_c,
        y_train=y_train_c,
        y_test=y_test_c,
        quantum_features=q_features,
        n_train_samples=q_train_n,
        n_test_samples=q_test_n,
        random_state=42,
    )

    assert X_train_q.shape == (q_train_n, q_features)
    assert X_test_q.shape == (q_test_n, q_features)
    assert len(selected_features) == q_features
    assert len(y_train_q) == q_train_n
    assert len(y_test_q) == q_test_n

    # Angle encoding values must be bounded within [0, pi]
    assert np.all(X_train_q >= -1e-5)
    assert np.all(X_train_q <= np.pi + 1e-5)
    assert np.all(X_test_q >= -1e-5)
    assert np.all(X_test_q <= np.pi + 1e-5)


def test_no_data_leakage(sample_dataframe):
    """
    Verify zero data leakage: train and test indices do not overlap,
    and scaler parameters are computed strictly from training split.
    """
    df = sample_dataframe.copy().reset_index(drop=True)
    df["sample_idx"] = np.arange(len(df))

    # Split features and labels
    X_raw = df[FEATURE_COLUMNS].to_numpy()
    y_raw = df[TARGET_COLUMN].to_numpy()

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    idx_train, idx_test = train_test_split(
        df["sample_idx"].to_numpy(),
        test_size=0.3,
        stratify=y_raw,
        random_state=42,
    )

    # 1. No index overlap between train and test
    overlap = set(idx_train).intersection(set(idx_test))
    assert len(overlap) == 0, f"Data leakage detected! Train/test index overlap: {overlap}"

    # 2. Scaler fit on train only
    train_raw = X_raw[idx_train]
    test_raw = X_raw[idx_test]

    scaler = StandardScaler().fit(train_raw)
    train_mean = train_raw.mean(axis=0)
    total_mean = X_raw.mean(axis=0)

    # Scaler mean must match training split mean, not combined mean
    np.testing.assert_allclose(scaler.mean_, train_mean)
    assert not np.allclose(scaler.mean_, total_mean), (
        "Scaler mean unexpectedly identical to total dataset mean (possible full dataset leak)."
    )
