# URK24CS1111
# QuantumSense - Synthetic Sensor Data Generation Module
# Simulates multi-sensor readings under normal and faulty operating conditions

"""
generate_data.py - Synthetic sensor data generator for QuantumSense.

Simulates 3 industrial sensors (Temperature, Pressure, Vibration) over sliding
time windows of WINDOW_SIZE=50 timesteps across 5 fault classes:
  - Class 0 (Normal): Gaussian variations around sensor nominal means
  - Class 1 (Drift): Correlated linear drift across readings (slope ~0.05-0.15)
  - Class 2 (Stuck): Signal freezes at an early value with tiny noise (std < 0.01)
  - Class 3 (Noisy): High-amplitude sensor noise (3-5x nominal standard deviation)
  - Class 4 (Spike): Intermittent impulse spikes (5-10x nominal standard deviation)

Extracts 15 time-domain statistical features per window (mean, std, min, max, slope)
and exports the dataset to SENSOR_DATA_CSV. Also saves representative waveform plots.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (
    DATA_DIR,
    RESULTS_DIR,
    SENSOR_DATA_CSV,
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
    ensure_dirs,
)


def simulate_sensor_window(
    fault_class: int,
    rng: np.random.RandomState,
    window_size: int = WINDOW_SIZE,
) -> Dict[str, np.ndarray]:
    """
    Simulate a single time window of raw readings for Temperature, Pressure,
    and Vibration sensors under the specified fault condition.

    Fault classes are physically correlated across the three sensors so that
    classes are distinct yet share realistic overlap at the boundaries.

    Parameters
    ----------
    fault_class : int
        Fault class index (0=Normal, 1=Drift, 2=Stuck, 3=Noisy, 4=Spike).
    rng : np.random.RandomState
        Random state instance for deterministic, sample-specific generation.
    window_size : int, optional
        Number of readings per window (default=WINDOW_SIZE).

    Returns
    -------
    dict
        Dictionary containing arrays of raw readings:
        {"temperature": np.ndarray, "pressure": np.ndarray, "vibration": np.ndarray}
    """
    t = np.arange(window_size, dtype=float)

    # Correlated background process noise shared across sensors
    shared_noise = rng.normal(0.0, 1.0, window_size)

    # Base nominal sensor signals with 15% cross-sensor covariance
    temp = rng.normal(TEMP_MEAN, TEMP_STD * 0.85, window_size) + shared_noise * (TEMP_STD * 0.15)
    pres = rng.normal(PRESSURE_MEAN, PRESSURE_STD * 0.85, window_size) + shared_noise * (PRESSURE_STD * 0.15)
    vib = rng.normal(VIBRATION_MEAN, VIBRATION_STD * 0.85, window_size) + shared_noise * (VIBRATION_STD * 0.15)

    if fault_class == 0:
        # Class 0: Normal operation
        # Natural operating point offsets between different equipment runs
        temp += rng.normal(0.0, TEMP_STD * 0.20)
        pres += rng.normal(0.0, PRESSURE_STD * 0.20)
        vib += rng.normal(0.0, VIBRATION_STD * 0.20)

    elif fault_class == 1:
        # Class 1: Drift fault
        # Gradual linear trend (e.g., thermal accumulation, filter clogging)
        # Temperature and pressure drift with slope ~0.05-0.15; vibration drifts proportionally
        drift_direction = rng.choice([-1.0, 1.0], p=[0.25, 0.75])
        temp_slope = drift_direction * rng.uniform(0.05, 0.15)
        pres_slope = drift_direction * rng.uniform(0.04, 0.12)
        vib_slope = drift_direction * rng.uniform(0.002, 0.008)

        temp += temp_slope * t
        pres += pres_slope * t
        vib += vib_slope * t

    elif fault_class == 2:
        # Class 2: Stuck sensor fault
        # Signal locks at an early reading with residual quantization/ADC noise (std < 0.01)
        stuck_idx = rng.randint(2, 8)
        tiny_noise_temp = rng.uniform(0.002, 0.008)
        tiny_noise_pres = rng.uniform(0.002, 0.008)
        tiny_noise_vib = rng.uniform(0.0005, 0.002)

        temp[stuck_idx:] = temp[stuck_idx] + rng.normal(0.0, tiny_noise_temp, window_size - stuck_idx)
        pres[stuck_idx:] = pres[stuck_idx] + rng.normal(0.0, tiny_noise_pres, window_size - stuck_idx)
        vib[stuck_idx:] = vib[stuck_idx] + rng.normal(0.0, tiny_noise_vib, window_size - stuck_idx)

    elif fault_class == 3:
        # Class 3: Noisy sensor fault
        # Severe broadband noise from loose mountings, electrical EMI, or cavitation
        # Standard deviation multiplied by 3-5x
        noise_factor = rng.uniform(3.0, 5.0)
        temp += rng.normal(0.0, TEMP_STD * (noise_factor - 1.0), window_size)
        pres += rng.normal(0.0, PRESSURE_STD * (noise_factor - 1.0), window_size)
        vib += rng.normal(0.0, VIBRATION_STD * (noise_factor - 1.0), window_size)

    elif fault_class == 4:
        # Class 4: Spike fault
        # Sudden impulses (water hammer, mechanical impact, power transient)
        # 3 to 7 sharp spikes with magnitude 5-10x nominal standard deviation
        num_spikes = rng.randint(3, 8)
        spike_indices = rng.choice(window_size, size=num_spikes, replace=False)
        for idx in spike_indices:
            spike_sign = rng.choice([-1.0, 1.0], p=[0.3, 0.7])
            spike_scale = rng.uniform(5.0, 10.0)
            temp[idx] += spike_sign * spike_scale * TEMP_STD
            pres[idx] += spike_sign * spike_scale * PRESSURE_STD * rng.uniform(0.8, 1.2)
            vib[idx] += spike_sign * spike_scale * VIBRATION_STD * rng.uniform(0.8, 1.2)

    else:
        raise ValueError(
            f"Invalid fault_class {fault_class}. Must be an integer in range [0, {NUM_CLASSES - 1}]."
        )

    return {
        "temperature": temp,
        "pressure": pres,
        "vibration": vib,
    }


def extract_window_features(
    temp: np.ndarray,
    pres: np.ndarray,
    vib: np.ndarray,
) -> Dict[str, float]:
    """
    Extract statistical time-domain features from raw sensor window readings.

    Features extracted for each sensor (15 total):
      - mean: average reading over the window
      - std: sample standard deviation (ddof=1)
      - min: minimum reading
      - max: maximum reading
      - slope: linear regression slope coefficient (rate of change per timestep)

    Parameters
    ----------
    temp : np.ndarray
        Raw temperature readings (WINDOW_SIZE elements).
    pres : np.ndarray
        Raw pressure readings (WINDOW_SIZE elements).
    vib : np.ndarray
        Raw vibration readings (WINDOW_SIZE elements).

    Returns
    -------
    dict
        Dictionary containing all 15 features matching FEATURE_COLUMNS order.
    """
    n = len(temp)
    t = np.arange(n, dtype=float)
    t_mean = (n - 1.0) / 2.0
    t_diff = t - t_mean
    t_denom = np.sum(t_diff ** 2)

    # Linear regression slopes: cov(t, signal) / var(t)
    temp_slope = float(np.sum(t_diff * (temp - np.mean(temp))) / t_denom)
    pres_slope = float(np.sum(t_diff * (pres - np.mean(pres))) / t_denom)
    vib_slope = float(np.sum(t_diff * (vib - np.mean(vib))) / t_denom)

    return {
        "temp_mean": float(np.mean(temp)),
        "temp_std": float(np.std(temp, ddof=1)),
        "temp_min": float(np.min(temp)),
        "temp_max": float(np.max(temp)),
        "temp_slope": temp_slope,

        "pres_mean": float(np.mean(pres)),
        "pres_std": float(np.std(pres, ddof=1)),
        "pres_min": float(np.min(pres)),
        "pres_max": float(np.max(pres)),
        "pres_slope": pres_slope,

        "vib_mean": float(np.mean(vib)),
        "vib_std": float(np.std(vib, ddof=1)),
        "vib_min": float(np.min(vib)),
        "vib_max": float(np.max(vib)),
        "vib_slope": vib_slope,
    }


def generate_single_sample(fault_class: int, seed: Optional[int] = None) -> Dict[str, Union[float, int, str]]:
    """
    Generate one sample's extracted features as a dictionary.
    Used by the web dashboard for real-time live sensor inference.

    Parameters
    ----------
    fault_class : int
        Fault class index (0 to NUM_CLASSES - 1).
    seed : int, optional
        Random seed for generation. If None, a non-deterministic seed is generated.

    Returns
    -------
    dict
        Dictionary with all 15 feature keys, plus TARGET_COLUMN and LABEL_COLUMN.
    """
    if fault_class not in FAULT_CLASSES:
        raise ValueError(
            f"Invalid fault_class {fault_class}. Must be one of {list(FAULT_CLASSES.keys())} "
            f"({CLASS_LABELS})."
        )

    if seed is None:
        seed = int(np.random.randint(0, 2_000_000_000))

    rng = np.random.RandomState(seed)
    sensor_data = simulate_sensor_window(fault_class, rng, WINDOW_SIZE)
    features = extract_window_features(
        sensor_data["temperature"],
        sensor_data["pressure"],
        sensor_data["vibration"],
    )
    features[TARGET_COLUMN] = int(fault_class)
    features[LABEL_COLUMN] = FAULT_CLASSES[fault_class]
    return features


def generate_single_sample_window(
    fault_class: int,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Union[float, int, str]], Dict[str, np.ndarray]]:
    """
    Generate both raw waveforms and extracted features for a single sample.
    Useful for dashboard time-series plotting alongside model predictions.

    Parameters
    ----------
    fault_class : int
        Fault class index (0 to NUM_CLASSES - 1).
    seed : int, optional
        Random seed for generation.

    Returns
    -------
    tuple of (dict, dict)
        (features_dict, raw_waveforms_dict)
    """
    if fault_class not in FAULT_CLASSES:
        raise ValueError(
            f"Invalid fault_class {fault_class}. Must be one of {list(FAULT_CLASSES.keys())}."
        )

    if seed is None:
        seed = int(np.random.randint(0, 2_000_000_000))

    rng = np.random.RandomState(seed)
    sensor_data = simulate_sensor_window(fault_class, rng, WINDOW_SIZE)
    features = extract_window_features(
        sensor_data["temperature"],
        sensor_data["pressure"],
        sensor_data["vibration"],
    )
    features[TARGET_COLUMN] = int(fault_class)
    features[LABEL_COLUMN] = FAULT_CLASSES[fault_class]
    return features, sensor_data


def plot_representative_waveforms(save_path: Optional[str] = None) -> str:
    """
    Generate and save a 5x3 multi-panel plot illustrating representative raw
    sensor waveforms for each of the 5 fault classes.

    Parameters
    ----------
    save_path : str, optional
        File path to save the generated PNG. Defaults to results/sample_waveforms.png.

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    ensure_dirs()
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "sample_waveforms.png")

    fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(15, 12), sharex=True)
    t = np.arange(WINDOW_SIZE)

    sensor_meta = [
        ("temperature", "Temperature ( degC)", "#d9534f"),
        ("pressure", "Pressure (kPa)", "#0275d8"),
        ("vibration", "Vibration (mm/s)", "#f0ad4e"),
    ]

    for c_idx in sorted(FAULT_CLASSES.keys()):
        class_label = FAULT_CLASSES[c_idx]
        # Fixed deterministic seed for representative waveform visualization
        rep_rng = np.random.RandomState(RANDOM_SEED + c_idx * 137)
        rep_data = simulate_sensor_window(c_idx, rep_rng, WINDOW_SIZE)

        for s_idx, (s_key, s_title, s_color) in enumerate(sensor_meta):
            ax = axes[c_idx, s_idx]
            ax.plot(t, rep_data[s_key], color=s_color, lw=1.6, label=f"Class {c_idx}")
            ax.grid(True, linestyle="--", alpha=0.5)

            if c_idx == 0:
                ax.set_title(s_title, fontsize=12, fontweight="bold", pad=8)
            if s_idx == 0:
                ax.set_ylabel(f"Class {c_idx}: {class_label}\n(Reading)", fontsize=10, fontweight="bold")
            if c_idx == NUM_CLASSES - 1:
                ax.set_xlabel("Time step", fontsize=10)

    fig.suptitle(
        "QuantumSense - Representative Sensor Waveforms by Fault Class",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Representative waveforms plot saved to: {save_path}")
    return save_path


def generate_dataset(
    samples_per_class: int = SAMPLES_PER_CLASS,
    window_size: int = WINDOW_SIZE,
    random_seed: int = RANDOM_SEED,
    save_csv: bool = True,
    save_plot: bool = True,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate the complete synthetic sensor fault detection dataset.

    Generates samples_per_class windows for each of the 5 fault classes.
    Uses a distinct random seed for each sample:
      rng = np.random.RandomState(random_seed + i)
    to guarantee reproducibility and ensure zero data leakage across train/test splits.

    Parameters
    ----------
    samples_per_class : int, optional
        Number of window samples per fault class (default=SAMPLES_PER_CLASS=200).
    window_size : int, optional
        Number of readings per window (default=WINDOW_SIZE=50).
    random_seed : int, optional
        Base seed for reproducibility (default=RANDOM_SEED=42).
    save_csv : bool, optional
        Whether to export the dataset to SENSOR_DATA_CSV (default=True).
    save_plot : bool, optional
        Whether to generate and save representative waveforms (default=True).

    Returns
    -------
    pd.DataFrame
        Complete dataset with 15 feature columns, fault_class, and fault_label.
    """
    ensure_dirs()
    print(f"[INFO] Generating sensor dataset with {NUM_CLASSES} classes, "
          f"{samples_per_class} samples/class ({NUM_CLASSES * samples_per_class} total samples)...")

    rows: List[Dict[str, Union[float, int, str]]] = []
    sample_counter = 0

    for fault_class in sorted(FAULT_CLASSES.keys()):
        for _ in range(samples_per_class):
            # Requirement 7: distinct reproducible seed per sample
            sample_rng = np.random.RandomState(random_seed + sample_counter)
            sensor_data = simulate_sensor_window(fault_class, sample_rng, window_size)
            features = extract_window_features(
                sensor_data["temperature"],
                sensor_data["pressure"],
                sensor_data["vibration"],
            )
            features[TARGET_COLUMN] = int(fault_class)
            features[LABEL_COLUMN] = FAULT_CLASSES[fault_class]
            rows.append(features)
            sample_counter += 1

    df = pd.DataFrame(rows)

    # Ensure column order matches specifications exactly
    ordered_cols = FEATURE_COLUMNS + [TARGET_COLUMN, LABEL_COLUMN]
    df = df[ordered_cols]

    # Validate generated data integrity
    nan_count = int(df.isna().sum().sum())
    inf_count = int(np.isinf(df[FEATURE_COLUMNS].to_numpy()).sum())

    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print("=" * 60)
    print(f"Total Rows:     {len(df)}")
    print(f"Total Columns:  {len(df.columns)} (15 features + 2 target/label)")
    print(f"Dimensions:     {df.shape}")
    print(f"NaN Values:     {nan_count}")
    print(f"Inf Values:     {inf_count}")
    print("\nClass Distribution:")
    for cls_idx, cls_name in FAULT_CLASSES.items():
        count = int((df[TARGET_COLUMN] == cls_idx).sum())
        pct = (count / len(df)) * 100.0
        print(f"  Class {cls_idx} ({cls_name:8s}): {count:4d} samples ({pct:5.1f}%)")
    print("=" * 60 + "\n")

    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"Dataset generation produced invalid numerical values! NaN: {nan_count}, Inf: {inf_count}"
        )

    # Save to CSV
    target_csv = output_path if output_path is not None else SENSOR_DATA_CSV
    if save_csv or output_path is not None:
        df.to_csv(target_csv, index=False)
        print(f"[INFO] Dataset successfully saved to: {target_csv}")

    # Generate representative waveforms plot
    if save_plot:
        plot_representative_waveforms()

    return df


# Compatibility aliases
generate_sensor_dataset = generate_dataset

try:
    from src.preprocess import validate_data
except ImportError:
    pass


if __name__ == "__main__":
    print("=" * 60)
    print("QuantumSense - Running Synthetic Sensor Data Generation")
    print("=" * 60)
    dataset = generate_dataset()
    print("[SUCCESS] Data generation pipeline executed successfully.")
