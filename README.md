# QuantumSense: Quantum-Enhanced Sensor Fault Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Qiskit 1.0+](https://img.shields.io/badge/Qiskit-1.0%2B-6929C4.svg)](https://qiskit.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Student ID](https://img.shields.io/badge/Student%20ID-URK24CS1111-brightgreen.svg)](#author)

> **QuantumSense** is an end-to-end quantum machine learning (QML) and cyber-physical systems project that investigates the application of quantum kernel methods for multi-sensor industrial fault diagnostics. By benchmarking a 4-qubit Quantum Support Vector Classifier (`ZZFeatureMap` kernel) against an optimized classical RBF Support Vector Classifier, the system evaluates how quantum Hilbert space encodings can isolate complex sensor failure modes. The complete suite provides physics-grounded synthetic telemetry generation, sliding-window feature engineering, rigorous empirical evaluation, and an interactive Flask web dashboard with real-time waveform monitoring and live fault injection.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#features)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation (Windows 11)](#installation)
- [Step-by-Step Usage Guide](#usage)
- [Expected Output Files](#expected-output-files)
- [Dashboard Screenshots](#dashboard-screenshots)
- [Troubleshooting & FAQs](#troubleshooting)
- [System Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Author & Acknowledgments](#author)

---

## Overview

Modern industrial cyber-physical systems (CPS), automated manufacturing cells, and critical power infrastructure rely upon continuous telemetry streams from multi-sensor arrays measuring temperature, pressure, and vibration. Unaddressed sensor failures—such as subtle calibration drift, frozen readings, high-frequency electromagnetic noise, or electrical surge spikes—lead to expensive equipment breakdowns, false emergency shutdowns, and significant safety hazards.

**QuantumSense** addresses this challenge by implementing an intelligent, adaptive diagnostic pipeline that systematically detects and classifies five distinct operational states (Normal, Drift, Stuck, Noisy, Spike). The project conducts a rigorous empirical comparison between a classical Support Vector Machine (RBF kernel) and a Quantum Support Vector Classifier (QSVC) utilizing Qiskit's `ZZFeatureMap` to project dimensionally reduced sensor dynamics into high-dimensional quantum Hilbert spaces. Complete with an interactive Flask web dashboard, QuantumSense bridges abstract quantum information science with practical industrial automation.

---

## Features

- **Multi-Sensor Stochastic Simulation:** Synthesizes realistic, physics-consistent time-series data for three industrial transducers:
  - **Temperature Sensor:** Nominal $75.0 \pm 2.0$ °C.
  - **Pressure Sensor:** Nominal $101.3 \pm 1.5$ kPa.
  - **Vibration Sensor:** Nominal $0.5 \pm 0.1$ mm/s.
- **5-Class Industrial Fault Taxonomy:** Mathematically models five core sensor operational modes:
  - `0 - Normal`: Stationary Gaussian stochastic noise around nominal baselines.
  - `1 - Drift`: Continuous progressive linear degradation ($\Delta y = m \cdot t$).
  - `2 - Stuck`: Transducer signal freeze clamped at a constant scalar ($\sigma \to 0$).
  - `3 - Noisy`: High-frequency variance amplification ($5\times$ standard deviation).
  - `4 - Spike`: Isolated extreme transient pulse anomalies ($> 4\sigma$).
- **Sliding-Window Feature Extraction:** Automatically computes 15 statistical and trend indicators per 50-step window (Mean, Standard Deviation, Min, Max, and Linear Trend Slope across all three sensor channels).
- **Optimized Classical Machine Learning Baseline:** Implements an RBF-kernel Support Vector Classifier (`C=10.0`, `gamma='scale'`) with standard normalization and stratified validation.
- **Quantum Machine Learning via QSVC:** Implements a 4-qubit Quantum Support Vector Classifier utilizing Qiskit's parameterized `ZZFeatureMap` (2 repetitions, linear entanglement topology) to calculate quantum state fidelity Gram matrices via statevector simulation.
- **Standardized Benchmark & Evaluation Suite:** Generates comprehensive classification reports, accuracy, macro precision/recall/F1 metrics, and high-resolution confusion matrix visualizations.
- **Interactive Full-Stack Web Dashboard:** A responsive Flask application featuring five dedicated operational views:
  - *Project Overview & KPIs:* System health and dataset summary.
  - *Live Sensor Monitor:* Continuous real-time waveform streaming and dynamic fault injection sandbox.
  - *Model Comparison Analytics:* Tabular benchmarks, confusion matrix inspections, and metric plots.
  - *Quantum Circuit Inspector:* Visual circuit breakdown of the `ZZFeatureMap` and interactive kernel matrix heatmap.
  - *Single-Sample Diagnostic Sandbox:* Interactive slider controls for real-time dual-model fault classification and automated maintenance guidance.
- **Automated Unit & Integration Test Suite:** Comprehensive test coverage via `pytest` validating data integrity, feature dimensions, model outputs, and web routes.

---

## Technologies Used

| Category | Technologies / Libraries |
| :--- | :--- |
| **Programming Language** | Python 3.10 / 3.11 |
| **Quantum Computing & QML** | Qiskit 1.0+, Qiskit Aer 0.13+, Qiskit Machine Learning 0.7+ |
| **Classical Machine Learning**| Scikit-learn 1.3+, Joblib 1.3+ |
| **Scientific Computing** | NumPy (<2.0), Pandas, SciPy |
| **Data Visualization** | Matplotlib 3.7+, Seaborn, Chart.js 4.0+ |
| **Web Framework & UI** | Flask 3.0+, Jinja2, HTML5, CSS3, Bootstrap 5 |
| **Testing & Quality Control**| Pytest 7.0+, Unittest |
| **Operating Environment** | Microsoft Windows 11 (PowerShell 7 / 5.1) |

---

## System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                     QUANTUMSENSE ARCHITECTURE                                     |
+---------------------------------------------------------------------------------------------------+

     [Multi-Sensor Simulation]
     +-------------------------------------------------------------+
     | Temperature Sensor (Mean: 75.0 °C, Std: 2.0 °C)             |
     | Pressure Sensor    (Mean: 101.3 kPa, Std: 1.5 kPa)          |
     | Vibration Sensor   (Mean: 0.50 mm/s, Std: 0.10 mm/s)        |
     | 5 Operational Modes: Normal, Drift, Stuck, Noisy, Spike     |
     | Total Records: 1,000 samples (200 samples/class)            |
     +------------------------------+------------------------------+
                                    |
                                    v
     [Windowing & Feature Extraction]
     +-------------------------------------------------------------+
     | Sliding Window Length: 50 steps                             |
     | Features (x3 Sensors): Mean, Std, Min, Max, Trend Slope     |
     | Total Dimensionality: 15 statistical features               |
     | Saved to: data/sensor_data.csv                              |
     +------------------------------+------------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
     [Classical Pipeline]            [Quantum Pipeline]
     +-----------------------------+ +-----------------------------+
     | 1. StandardScaler (15 feats)| | 1. PCA Reduction (4 feats)  |
     | 2. Stratified 75/25 Split   | | 2. MinMax Scale to [0, 2π]  |
     | 3. SVC (RBF, C=10.0, scale) | | 3. 4-Qubit ZZFeatureMap     |
     | 4. Convex Quadratic Prog.   | |    (reps=2, linear entangle)|
     | 5. Fast classical inference | | 4. Fidelity Quantum Kernel  |
     | Output:                     | | 5. QSVC Dual Optimizer      |
     | classical_svm_model.joblib  | | Output:                     |
     |                             | | quantum_results.joblib      |
     +--------------+--------------+ +--------------+--------------+
                    |                               |
                    +---------------+---------------+
                                    |
                                    v
     [Evaluation & Benchmarking Engine]
     +-------------------------------------------------------------+
     | Metrics: Accuracy, Precision, Recall, Macro F1, Latency     |
     | Confusion Matrix Generation & Performance Serialization     |
     | Output: results/metrics.json, predictions.csv, PNG plots    |
     +------------------------------+------------------------------+
                                    |
                                    v
     [Interactive Flask Web Diagnostic Dashboard (Port 5000)]
     +-------------------------------------------------------------+
     | Tab 1: Project Overview & Executive Metrics                 |
     | Tab 2: Live Telemetry Waveforms & Fault Injection Sandbox   |
     | Tab 3: Classical vs Quantum Performance Comparative Matrix  |
     | Tab 4: Quantum Circuit & Fidelity Kernel Matrix Heatmap      |
     | Tab 5: Single-Sample Live Fault Diagnostics & Remediation   |
     +-------------------------------------------------------------+
```

---

## Project Structure

```
QuantumSense/
├── data/
│   └── sensor_data.csv                   # Generated 15-feature multi-sensor dataset
├── models/
│   ├── classical_model.py                # Classical RBF SVM training and serialization
│   └── quantum_model.py                  # QSVC training via Qiskit ZZFeatureMap
├── results/
│   ├── classical_scaler.joblib           # Fitted StandardScaler for classical model
│   ├── classical_svm_model.joblib        # Trained classical SVM model artifact
│   ├── quantum_scaler.joblib             # Fitted MinMaxScaler for quantum pipeline
│   ├── quantum_results.joblib            # Trained quantum kernel results & metadata
│   ├── metrics.json                      # Comprehensive evaluation metrics (JSON)
│   ├── predictions.csv                   # Test set predictions for both models
│   ├── model_comparison.png              # Comparative performance bar chart
│   ├── confusion_matrix_classical.png    # Classical SVM confusion matrix plot
│   └── confusion_matrix_quantum.png      # Quantum QSVC confusion matrix plot
├── src/
│   ├── __init__.py                       # Package initializer
│   ├── config.py                         # Single source of truth configuration
│   ├── generate_data.py                  # Sensor simulation and feature extraction
│   └── evaluate.py                       # Evaluation, metrics generation, and plotting
├── dashboard/
│   ├── app.py                            # Flask server and API endpoints
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css                 # Custom industrial dark/light styling
│   │   └── js/
│   │       ├── main.js                   # Client-side UI controls and AJAX handlers
│   │       └── charts.js                 # Chart.js waveform and metric visualizations
│   └── templates/
│       ├── base.html                     # Base HTML template with navigation
│       ├── index.html                    # Tab 1: Project Overview & KPIs
│       ├── monitor.html                  # Tab 2: Live Telemetry & Fault Injection
│       ├── evaluation.html               # Tab 3: Model Benchmarking Analytics
│       ├── quantum.html                  # Tab 4: Quantum Circuit & Kernel Heatmap
│       └── predict.html                  # Tab 5: Real-time Fault Predictor
├── tests/
│   ├── __init__.py                       # Package initializer
│   ├── test_data.py                      # Dataset schema and statistics unit tests
│   ├── test_models.py                    # Classical and quantum model tests
│   └── test_dashboard.py                 # Flask route and API endpoint tests
├── PRESENTATION_OUTLINE.md               # 12-slide presentation outline & 30 viva Q&As
├── README.md                             # Project documentation (this file)
└── requirements.txt                      # Pinned Python package dependencies
```

---

## Installation

Follow these step-by-step PowerShell instructions on Windows 11 to set up your local development environment:

### Step 1: Open PowerShell as User
Launch Windows PowerShell or Windows Terminal.

### Step 2: Navigate to Project Directory
```powershell
cd "d:\Downloads\quantum computing project\QuantumSense"
```

### Step 3: Configure PowerShell Script Execution Policy (If Needed)
If your system restricts running PowerShell virtual environment scripts, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Create a Dedicated Python Virtual Environment
```powershell
python -m venv venv
```

### Step 5: Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(Your command prompt will display `(venv)` upon successful activation).*

### Step 6: Upgrade Pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

Execute the complete end-to-end QuantumSense workflow in sequence:

### Step 1: Generate Multi-Sensor Dataset
Generates physics-modeled sensor readings across 5 fault classes and extracts 15 sliding-window features:
```powershell
python src/generate_data.py
```
*Output: `data/sensor_data.csv` (1,000 samples $\times$ 17 columns).*

### Step 2: Train Classical SVM Baseline
Normalizes features and fits a Support Vector Classifier with an RBF kernel:
```powershell
python models/classical_model.py
```
*Output: `results/classical_svm_model.joblib` and `results/classical_scaler.joblib` (Executes in < 3 seconds).*

### Step 3: Train Quantum Support Vector Classifier (QSVC)
> **Note:** Evaluating quantum statevector fidelities on a classical CPU simulator computes pairwise inner products ($O(N^2)$). This step intentionally trains on a compact, representative subset ($N_{\text{train}}=80, N_{\text{test}}=40$) and typically requires **5 to 15 minutes** depending on CPU performance. Please allow the process to finish.
```powershell
python models/quantum_model.py
```
*Output: `results/quantum_results.joblib` and `results/quantum_scaler.joblib`.*

### Step 4: Run Comparative Evaluation & Generate Plots
Calculates accuracy, precision, recall, macro F1-score, and creates confusion matrices and comparison charts:
```powershell
python src/evaluate.py
```
*Output: `results/metrics.json`, `results/predictions.csv`, and PNG chart plots.*

### Step 5: Start Diagnostic Web Dashboard
Launches the interactive Flask diagnostic server:
```powershell
python dashboard/app.py
```
Open your web browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### Step 6: Run Automated Test Suite
Verify that all unit and integration tests pass:
```powershell
python -m pytest tests/ -v
```

---

## Expected Output Files

Upon running the complete pipeline, the following files are produced:

| Directory | Output Filename | Description |
| :--- | :--- | :--- |
| `data/` | `sensor_data.csv` | 1,000 processed samples with 15 engineered features, `fault_class` (0–4), and `fault_label`. |
| `results/` | `classical_svm_model.joblib` | Trained classical scikit-learn `SVC` model artifact. |
| `results/` | `classical_scaler.joblib` | Fitted `StandardScaler` used for classical feature normalization. |
| `results/` | `quantum_results.joblib` | Pickled dictionary containing quantum predictions, test indices, and kernel metadata. |
| `results/` | `quantum_scaler.joblib` | Fitted `MinMaxScaler` mapping features to $[0, 2\pi]$ for angle encoding. |
| `results/` | `metrics.json` | Detailed JSON file containing accuracy, precision, recall, and F1-score for both models. |
| `results/` | `predictions.csv` | Test dataset records with ground truth, classical predictions, and quantum predictions. |
| `results/` | `model_comparison.png` | Grouped bar chart comparing Accuracy, Precision, Recall, and F1 between Classical and Quantum. |
| `results/` | `confusion_matrix_classical.png`| Heatmap confusion matrix of classical SVM predictions across all 5 classes. |
| `results/` | `confusion_matrix_quantum.png`  | Heatmap confusion matrix of quantum QSVC predictions across all 5 classes. |

---

## Dashboard Screenshots

The QuantumSense dashboard provides an intuitive, industrial-grade monitoring interface structured across five operational views:

> `[Screenshots will be added after running the dashboard]`

- **Tab 1: Executive Overview:** Comprehensive KPI summary tiles, architecture breakdown, sensor distribution, and active system status.
- **Tab 2: Live Sensor Monitor:** Dynamic multi-channel time-series charts displaying Temperature, Pressure, and Vibration waveforms with live interactive fault injection.
- **Tab 3: Model Evaluation Analytics:** Side-by-side performance comparison tables, interactive confusion matrices, and precision-recall trade-off graphs.
- **Tab 4: Quantum Circuit & Kernel Explorer:** Interactive schematic of the 4-qubit `ZZFeatureMap` circuit with parameterized $R_z$ rotations and quantum Gram matrix fidelity heatmaps.
- **Tab 5: Single-Sample Fault Predictor:** Operator sandbox with 15 parameter sliders for instant multi-model inference, confidence scores, and automated maintenance advisories.

---

## Troubleshooting

Common setup or runtime issues and their corresponding resolutions:

#### 1. `python` is not recognized as an internal or external command
- **Cause:** Python is not added to the system `PATH` environment variable.
- **Fix:** Reinstall Python from [python.org](https://www.python.org/) and ensure **"Add python.exe to PATH"** is checked, or add `C:\Users\<User>\AppData\Local\Programs\Python\Python311` manually to your System Environment Variables.

#### 2. PowerShell Script Execution Blocked (`Activate.ps1 cannot be loaded`)
- **Cause:** Windows PowerShell execution policy restricts running unsigned local scripts.
- **Fix:** Run this command in PowerShell:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

#### 3. Qiskit Installation Fails or Throws Dependency Errors
- **Cause:** Incompatible build dependencies or conflicting compiler versions.
- **Fix:** Install the recommended stable release directly:
  ```powershell
  pip install qiskit==1.2.0 qiskit-aer==0.15.1 qiskit-machine-learning==0.7.2
  ```

#### 4. NumPy Version Conflict (`numpy>=2.0` breaking C-extensions)
- **Cause:** NumPy 2.0+ introduces breaking ABI changes with certain compiled wheels.
- **Fix:** Enforce NumPy 1.x:
  ```powershell
  pip install "numpy>=1.24,<2.0"
  ```

#### 5. Quantum Model Training is Slow (`quantum_model.py` taking over 10 minutes)
- **Cause:** Expected behavior. Classical CPU simulation of quantum state fidelities across $N$ samples requires $O(N^2)$ circuit executions.
- **Fix:** Let the training complete, or adjust `QUANTUM_TRAIN_SAMPLES = 40` and `QUANTUM_TEST_SAMPLES = 20` in `src/config.py` for faster testing.

#### 6. Flask Port Conflict (`Address already in use: Port 5000`)
- **Cause:** Another background service or previous Flask instance is occupying port 5000.
- **Fix:** Change `FLASK_PORT = 5001` (or another free port) in `src/config.py`, or kill the occupying process:
  ```powershell
  Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess | Stop-Process -Force
  ```

#### 7. ModuleNotFoundError: No module named 'src' or 'models'
- **Cause:** Running scripts from a subfolder rather than the project root.
- **Fix:** Always execute commands from the `QuantumSense/` root folder:
  ```powershell
  cd "d:\Downloads\quantum computing project\QuantumSense"
  python src/generate_data.py
  ```

#### 8. FileNotFoundError: `data/sensor_data.csv` missing
- **Cause:** Attempting to train models before generating the dataset.
- **Fix:** Run `python src/generate_data.py` first to generate the dataset.

#### 9. FileNotFoundError: `results/classical_svm_model.joblib` or `quantum_results.joblib` missing
- **Cause:** Attempting to evaluate or launch the dashboard before running training.
- **Fix:** Execute `python models/classical_model.py` and `python models/quantum_model.py` before running `python src/evaluate.py` or launching the dashboard.

---

## Limitations

An honest assessment of the system's current engineering boundaries:

- **Synthetic Telemetry Basis:** The dataset is generated through stochastic mathematical formulations rather than physical industrial testbeds. Real industrial sensors encounter complex non-stationary environmental dynamics, physical wear, and unmodeled noise that may require transfer learning.
- **Classical Simulation Bottleneck:** Multi-qubit circuit simulation on classical computers scales exponentially with qubit count ($O(2^n)$). This computational bottleneck limits the quantum model to a 4-qubit feature representation and 80 training samples for tractable execution on desktop hardware.
- **Quadratic Gram Matrix Scaling:** Quantum kernel methods require computing all $O(N^2)$ pairwise inner products between training samples. While feasible on small datasets, scaling to millions of streaming data points requires batching, Nyström kernel approximations, or Variational Quantum Classifiers (VQC).
- **Inference Latency Trade-Off:** While classical SVM yields sub-millisecond inference suitable for hard real-time safety shutoffs, statevector-simulated quantum inference takes several seconds per sample on classical CPUs.

---

## Future Enhancements

- **Real Quantum Hardware Execution:** Deploy the `ZZFeatureMap` circuits on real IBM Quantum superconducting hardware backends (such as IBM Quantum Eagle or Heron) via the Qiskit Runtime Service with Zero-Noise Extrapolation (ZNE).
- **Variational Quantum Classifiers (VQC / QNN):** Implement parameterized Quantum Neural Networks optimized via the parameter-shift rule to achieve linear $O(N)$ inference complexity.
- **Physical Sensor Testbed Integration:** Validate and fine-tune models against real-world industrial benchmarks, such as the NASA C-MAPSS Turbofan Degradation Dataset and Case Western Reserve University (CWRU) Bearing Dataset.
- **Edge-to-Cloud Quantum Hybrid Architecture:** Develop an industrial edge client that extracts 15 sliding-window features on an embedded gateway (e.g., Raspberry Pi / NVIDIA Jetson) and asynchronously offloads quantum kernel evaluations to quantum cloud coprocessors.
- **Adaptive Dynamic Windowing:** Implement reinforcement learning or entropy-driven adaptive window sizing to dynamically optimize temporal resolution during rapid transient fault events.

---

## License

This project is open-source and distributed under the **[MIT License](https://opensource.org/licenses/MIT)**.

```
MIT License

Copyright (c) 2026-2027 URK24CS1111

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Author

- **Student Name / Register Number:** URK24CS1111
- **Degree:** Bachelor of Technology in Computer Science and Engineering (B.Tech CSE)
- **Department:** Department of Computer Science and Engineering
- **Project Domain:** Cyber-Physical Systems (CPS) & Quantum Machine Learning (QML)
- **Institution:** B.Tech Final Year Capstone Project
