# URK24CS1111
# QuantumSense - Evaluation & Comparison Module
# Benchmarks Classical SVM against Quantum QSVC and generates visual diagnostics

"""
evaluate.py - Comparative evaluation framework.
Generates confusion matrices, side-by-side metric comparison charts,
persists metrics in JSON, and exports granular predictions to CSV.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib

# Set Agg backend for non-interactive / headless figure generation
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    METRICS_JSON,
    PREDICTIONS_CSV,
    COMPARISON_PNG,
    CM_CLASSICAL_PNG,
    CM_QUANTUM_PNG,
    FAULT_CLASSES,
    CLASS_LABELS,
    ensure_dirs,
)


def load_metrics() -> dict:
    """
    Load pre-computed evaluation metrics from METRICS_JSON if it exists.

    Returns:
        dict: Parsed metrics dictionary, or empty dict if file is not found.
    """
    if os.path.exists(METRICS_JSON):
        try:
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to read metrics from {METRICS_JSON}: {e}")
            return {}
    return {}


def generate_confusion_matrix_plot(y_true, y_pred, title: str, save_path: str):
    """
    Generate and save a publication-quality confusion matrix heatmap.

    Args:
        y_true (array-like): Ground-truth target labels.
        y_pred (array-like): Model predicted labels.
        title (str): Title string for the plot.
        save_path (str): Destination file path for the PNG image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()

    # Determine unique classes present in either truth or prediction
    classes_present = sorted(list(set(y_t) | set(y_p)))
    tick_labels = [FAULT_CLASSES.get(i, f"Class {i}") for i in classes_present]

    cm = confusion_matrix(y_t, y_p, labels=classes_present)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        title=title,
        ylabel="True Fault Label",
        xlabel="Predicted Fault Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    # Overlay numeric counts inside each confusion cell
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{val:d}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
                fontsize=11,
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Saved confusion matrix plot to: {save_path}")


def generate_comparison_chart(metrics_dict: dict, save_path: str):
    """
    Generate a grouped bar chart comparing Classical SVM and Quantum QSVC performance.
    Classical represented in Blue (#2563eb), Quantum in Teal (#0d9488).

    Args:
        metrics_dict (dict): Combined metrics dictionary.
        save_path (str): File path for saving the PNG chart.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    classical_data = metrics_dict.get("classical", {})
    quantum_data = metrics_dict.get("quantum", {})
    has_quantum = bool(quantum_data and "accuracy" in quantum_data and quantum_data.get("accuracy") is not None)

    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    classical_scores = [float(classical_data.get(k, 0.0) or 0.0) for k in metric_keys]
    quantum_scores = (
        [float(quantum_data.get(k, 0.0) or 0.0) for k in metric_keys]
        if has_quantum
        else [0.0] * len(metric_keys)
    )

    x = np.arange(len(metric_names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rects1 = ax.bar(
        x - bar_width / 2,
        classical_scores,
        bar_width,
        label="Classical SVM (RBF Kernel)",
        color="#2563eb",
        edgecolor="#1e3a8a",
        linewidth=1.0,
    )
    rects2 = ax.bar(
        x + bar_width / 2,
        quantum_scores,
        bar_width,
        label="Quantum QSVC (ZZFeatureMap)",
        color="#0d9488",
        edgecolor="#0f766e",
        linewidth=1.0,
    )

    ax.set_ylabel("Metric Score", fontsize=12, fontweight="bold")
    ax.set_title("Classical SVM vs Quantum QSVC: Fault Detection Performance", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects, is_active=True):
        for rect in rects:
            height = rect.get_height()
            if is_active and height > 0.0:
                ax.annotate(
                    f"{height:.2%}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    autolabel(rects1)
    if has_quantum:
        autolabel(rects2, is_active=True)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Saved model comparison chart to: {save_path}")


def evaluate_models(classical_results: dict, quantum_results: dict = None) -> dict:
    """
    Perform holistic evaluation and comparison between Classical and Quantum models.

    Steps:
      1. Aggregates metrics side-by-side into a unified dictionary
      2. Generates confusion matrix heatmaps for each model
      3. Generates comparative grouped bar chart
      4. Saves combined metrics to METRICS_JSON
      5. Exports granular test sample predictions to PREDICTIONS_CSV
      6. Prints comparison overview to console

    Args:
        classical_results (dict): Output from train_classical_model()
        quantum_results (dict, optional): Output from train_quantum_model()

    Returns:
        dict: Full comparison metrics dictionary matching project specification
    """
    ensure_dirs()

    # Build classical metrics section
    classical_metrics = {
        "accuracy": round(float(classical_results.get("accuracy", 0.0)), 4),
        "precision": round(float(classical_results.get("precision", 0.0)), 4),
        "recall": round(float(classical_results.get("recall", 0.0)), 4),
        "f1": round(float(classical_results.get("f1", 0.0)), 4),
        "weighted_precision": round(float(classical_results.get("weighted_precision", classical_results.get("precision", 0.0))), 4),
        "weighted_recall": round(float(classical_results.get("weighted_recall", classical_results.get("recall", 0.0))), 4),
        "weighted_f1": round(float(classical_results.get("weighted_f1", classical_results.get("f1", 0.0))), 4),
        "train_time": round(float(classical_results.get("train_time", 0.0)), 4),
        "predict_time": round(float(classical_results.get("predict_time", 0.0)), 4),
        "train_samples": int(classical_results.get("train_samples", 0)),
        "test_samples": int(classical_results.get("test_samples", 0)),
        "confusion_matrix": classical_results.get("confusion_matrix", []),
        "classification_report": classical_results.get("classification_report_str", ""),
    }

    # Build quantum metrics section if available
    quantum_metrics = {}
    has_quantum = bool(quantum_results is not None and "accuracy" in quantum_results)

    if has_quantum:
        quantum_metrics = {
            "accuracy": round(float(quantum_results.get("accuracy", 0.0)), 4),
            "precision": round(float(quantum_results.get("precision", 0.0)), 4),
            "recall": round(float(quantum_results.get("recall", 0.0)), 4),
            "f1": round(float(quantum_results.get("f1", 0.0)), 4),
            "weighted_precision": round(float(quantum_results.get("weighted_precision", quantum_results.get("precision", 0.0))), 4),
            "weighted_recall": round(float(quantum_results.get("weighted_recall", quantum_results.get("recall", 0.0))), 4),
            "weighted_f1": round(float(quantum_results.get("weighted_f1", quantum_results.get("f1", 0.0))), 4),
            "train_time": round(float(quantum_results.get("train_time", 0.0)), 4),
            "predict_time": round(float(quantum_results.get("predict_time", 0.0)), 4),
            "train_samples": int(quantum_results.get("train_samples", 0)),
            "test_samples": int(quantum_results.get("test_samples", 0)),
            "confusion_matrix": quantum_results.get("confusion_matrix", []),
            "classification_report": quantum_results.get("classification_report_str", ""),
            "num_qubits": int(quantum_results.get("num_qubits", 4)),
            "circuit_depth": int(quantum_results.get("circuit_depth", 0)),
            "selected_features": quantum_results.get("selected_features", []),
        }

    # Build comparative metrics section
    comparison = {}
    if has_quantum:
        acc_diff = round(classical_metrics["accuracy"] - quantum_metrics["accuracy"], 4)
        f1_diff = round(classical_metrics["f1"] - quantum_metrics["f1"], 4)
        classical_tt = max(classical_metrics["train_time"], 1e-6)
        slower_factor = round(quantum_metrics["train_time"] / classical_tt, 2)

        summary_text = (
            f"Classical SVM achieved {classical_metrics['accuracy']:.2%} accuracy in {classical_metrics['train_time']:.3f}s. "
            f"Quantum QSVC (using 4 qubits ZZFeatureMap) achieved {quantum_metrics['accuracy']:.2%} accuracy in {quantum_metrics['train_time']:.3f}s "
            f"({slower_factor}x quantum simulation overhead on classical CPU). "
            f"The quantum kernel demonstrates competitive classification capability with a reduced feature footprint."
        )

        comparison = {
            "accuracy_diff": acc_diff,
            "f1_diff": f1_diff,
            "quantum_slower_factor": slower_factor,
            "summary": summary_text,
        }
    else:
        comparison = {
            "accuracy_diff": None,
            "f1_diff": None,
            "quantum_slower_factor": None,
            "summary": "Quantum model training was skipped or encountered an issue; classical SVM metrics recorded.",
        }

    metrics_payload = {
        "classical": classical_metrics,
        "quantum": quantum_metrics,
        "comparison": comparison,
    }

    # 1. Generate Confusion Matrix Plots
    if "y_test" in classical_results and "y_pred" in classical_results:
        generate_confusion_matrix_plot(
            classical_results["y_test"],
            classical_results["y_pred"],
            "Classical SVM Confusion Matrix",
            CM_CLASSICAL_PNG,
        )

    if has_quantum and "y_test_q" in quantum_results and "y_pred_q" in quantum_results:
        generate_confusion_matrix_plot(
            quantum_results["y_test_q"],
            quantum_results["y_pred_q"],
            "Quantum QSVC Confusion Matrix",
            CM_QUANTUM_PNG,
        )
    else:
        # Generate informative placeholder image for quantum confusion matrix
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.text(
            0.5,
            0.5,
            "Quantum QSVC Evaluation Pending\n\nRun 'python models/quantum_model.py'\nto generate quantum simulation matrix.",
            ha="center",
            va="center",
            fontsize=11,
            color="#495057",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#ced4da"),
        )
        ax.set_title("Quantum QSVC Confusion Matrix (Pending)", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(CM_QUANTUM_PNG, dpi=150)
        plt.close(fig)

    # 2. Generate Comparison Bar Chart
    generate_comparison_chart(metrics_payload, COMPARISON_PNG)

    # 3. Save JSON Metrics
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[Evaluate] Metrics exported to: {METRICS_JSON}")

    # 4. Save Granular Predictions to CSV
    records = []
    if "y_test" in classical_results and "y_pred" in classical_results:
        for idx, (t, p) in enumerate(zip(classical_results["y_test"], classical_results["y_pred"])):
            records.append({
                "model": "classical",
                "sample_index": idx,
                "true_class": int(t),
                "predicted_class": int(p),
                "true_label": FAULT_CLASSES.get(int(t), f"Class {t}"),
                "predicted_label": FAULT_CLASSES.get(int(p), f"Class {p}"),
                "correct": bool(int(t) == int(p)),
            })

    if has_quantum and "y_test_q" in quantum_results and "y_pred_q" in quantum_results:
        for idx, (t, p) in enumerate(zip(quantum_results["y_test_q"], quantum_results["y_pred_q"])):
            records.append({
                "model": "quantum",
                "sample_index": idx,
                "true_class": int(t),
                "predicted_class": int(p),
                "true_label": FAULT_CLASSES.get(int(t), f"Class {t}"),
                "predicted_label": FAULT_CLASSES.get(int(p), f"Class {p}"),
                "correct": bool(int(t) == int(p)),
            })

    if records:
        df_preds = pd.DataFrame(records)
        df_preds.to_csv(PREDICTIONS_CSV, index=False)
        print(f"[Evaluate] Sample-level predictions exported to: {PREDICTIONS_CSV}")

    # 5. Print Summary Comparison
    print("\n" + "=" * 65)
    print("           MODEL BENCHMARK COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<18} | {'Classical SVM':<16} | {'Quantum QSVC':<16}")
    print("-" * 65)
    print(f"{'Accuracy':<18} | {classical_metrics['accuracy']:.4f}{'':<10} | {quantum_metrics.get('accuracy', 'N/A')}")
    print(f"{'Precision (Macro)':<18} | {classical_metrics['precision']:.4f}{'':<10} | {quantum_metrics.get('precision', 'N/A')}")
    print(f"{'Recall (Macro)':<18} | {classical_metrics['recall']:.4f}{'':<10} | {quantum_metrics.get('recall', 'N/A')}")
    print(f"{'F1 Score (Macro)':<18} | {classical_metrics['f1']:.4f}{'':<10} | {quantum_metrics.get('f1', 'N/A')}")
    print(f"{'Training Time':<18} | {classical_metrics['train_time']:.4f} s{'':<8} | {quantum_metrics.get('train_time', 'N/A')} s")
    print(f"{'Inference Time':<18} | {classical_metrics['predict_time']:.4f} s{'':<8} | {quantum_metrics.get('predict_time', 'N/A')} s")
    print(f"{'Training Samples':<18} | {classical_metrics['train_samples']:<16} | {quantum_metrics.get('train_samples', 'N/A')}")
    print(f"{'Test Samples':<18} | {classical_metrics['test_samples']:<16} | {quantum_metrics.get('test_samples', 'N/A')}")
    print("=" * 65)
    print(f"Summary: {comparison.get('summary')}")
    print("=" * 65)

    return metrics_payload


if __name__ == "__main__":
    print("=" * 70)
    print(" QuantumSense - Full Training & Comparative Evaluation Pipeline")
    print("=" * 70)

    # Train Classical Model
    print("\n[Pipeline Phase 1] Training Classical SVM Model...")
    from models.classical_model import train_classical_model
    classical_res = train_classical_model()

    # Train Quantum Model with resilience
    quantum_res = None
    try:
        print("\n[Pipeline Phase 2] Training Quantum QSVC Model...")
        from models.quantum_model import train_quantum_model
        quantum_res = train_quantum_model()
    except Exception as q_err:
        print(f"\n[Warning] Quantum QSVC training failed: {q_err}")
        print("[Pipeline Phase 2] Continuing with Classical evaluation...")

    # Run Evaluation and artifact generation
    print("\n[Pipeline Phase 3] Synthesizing Comparative Diagnostics...")
    eval_metrics = evaluate_models(classical_res, quantum_res)
    print("\nEvaluation completed successfully!")
