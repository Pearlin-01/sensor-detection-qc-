# URK24CS1111
"""
Script to generate a professional 9-slide PowerPoint presentation
for QuantumSense: Quantum-Enhanced Sensor Fault Detection.
Uses python-pptx with custom styling, dark-themed industrial aesthetic,
clean typography, and structured bullet layouts.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck(output_path="QuantumSense_Presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette: Dark Industrial IoT
    DARK_BG = RGBColor(10, 15, 30)       # #0a0f1e
    CARD_BG = RGBColor(22, 33, 62)       # #16213e
    ACCENT_TEAL = RGBColor(0, 180, 216)  # #00b4d8
    ACCENT_BLUE = RGBColor(72, 202, 228) # #48cae4
    TEXT_WHITE = RGBColor(240, 244, 248) # #f0f4f8
    TEXT_MUTED = RGBColor(160, 174, 192) # #a0aec0
    ACCENT_GREEN = RGBColor(0, 200, 83)  # #00c853
    ACCENT_ORANGE = RGBColor(255, 160, 0)# #ffa000

    def add_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = DARK_BG
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, category_text="QUANTUMSENSE • B.TECH CSE"):
        # Category Tracker
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.name = "Arial"
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_TEAL

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = "Arial"
        p_title.font.size = Pt(26)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE

    def add_card(slide, left, top, width, height, title, bullet_points, highlight_color=None):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = highlight_color if highlight_color else ACCENT_TEAL
        card.line.width = Pt(1.5)

        tx_box = slide.shapes.add_textbox(Inches(left + 0.25), Inches(top + 0.2), Inches(width - 0.5), Inches(height - 0.4))
        tf = tx_box.text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.name = "Arial"
        p0.font.size = Pt(18)
        p0.font.bold = True
        p0.font.color.rgb = highlight_color if highlight_color else ACCENT_BLUE
        p0.space_after = Pt(12)

        for pt in bullet_points:
            p = tf.add_paragraph()
            p.text = f"•  {pt}"
            p.font.name = "Calibri"
            p.font.size = Pt(14)
            p.font.color.rgb = TEXT_WHITE
            p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    add_background(s1)

    t_box = s1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(10.9), Inches(3.8))
    tf1 = t_box.text_frame
    tf1.word_wrap = True

    p_badge = tf1.paragraphs[0]
    p_badge.text = "⚡ INDUSTRIAL IOT & QUANTUM MACHINE LEARNING"
    p_badge.font.name = "Arial"
    p_badge.font.size = Pt(14)
    p_badge.font.bold = True
    p_badge.font.color.rgb = ACCENT_TEAL
    p_badge.space_after = Pt(14)

    p_main = tf1.add_paragraph()
    p_main.text = "QuantumSense"
    p_main.font.name = "Arial"
    p_main.font.size = Pt(46)
    p_main.font.bold = True
    p_main.font.color.rgb = TEXT_WHITE
    p_main.space_after = Pt(8)

    p_sub = tf1.add_paragraph()
    p_sub.text = "Quantum-Enhanced Sensor Fault Detection Using Quantum Machine Learning"
    p_sub.font.name = "Calibri"
    p_sub.font.size = Pt(22)
    p_sub.font.color.rgb = ACCENT_BLUE
    p_sub.space_after = Pt(28)

    p_meta = tf1.add_paragraph()
    p_meta.text = "Author Registration ID: URK24CS1111  |  B.Tech Computer Science & Engineering  |  Academic Year 2024–2025"
    p_meta.font.name = "Calibri"
    p_meta.font.size = Pt(14)
    p_meta.font.color.rgb = TEXT_MUTED

    # -------------------------------------------------------------
    # SLIDE 2: Problem Statement & Motivation
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_background(s2)
    add_header(s2, "Industrial Problem Statement & Project Motivation")

    add_card(s2, 0.8, 1.8, 5.6, 5.0, "Challenges in Sensor Fault Detection", [
        "Modern industrial automation relies on multi-sensor telemetry (temperature, pressure, vibration).",
        "Sensor failures cause costly unexpected downtime, equipment damage, and environmental hazards.",
        "Traditional static thresholds fail to catch non-linear correlations, slow monotonic drifts, and intermittent transients.",
        "Classical ML models can suffer from dimensionality hurdles and high variance in boundary detection."
    ], ACCENT_ORANGE)

    add_card(s2, 6.8, 1.8, 5.7, 5.0, "The QuantumSense Solution", [
        "Investigates Quantum Machine Learning (QML) for non-linear pattern recognition in industrial sensors.",
        "Applies Quantum Feature Maps (ZZFeatureMap) to encode multi-modal sensor signals into high-dimensional Hilbert space.",
        "Employs Quantum Support Vector Classifiers (QSVC) with Fidelity Quantum Kernels.",
        "Provides rigorous empirical side-by-side benchmarking against classical Support Vector Machines (SVM)."
    ], ACCENT_TEAL)

    # -------------------------------------------------------------
    # SLIDE 3: System Architecture & End-to-End Pipeline
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_background(s3)
    add_header(s3, "System Architecture & Processing Pipeline")

    add_card(s3, 0.8, 1.8, 3.6, 5.0, "1. Ingestion & Preprocessing", [
        "Synthetic telemetry simulation (PT100, Piezo, MEMS).",
        "Custom CSV dataset upload lab with auto-column mapping.",
        "Mode A: Precomputed statistical features.",
        "Mode B: Raw sliding-window feature engineering (W=50).",
        "Zero data leakage: scaling fit strictly on train split."
    ])

    add_card(s3, 4.8, 1.8, 3.7, 5.0, "2. Dual Model Engines", [
        "Classical SVM: Multi-class RBF kernel baseline (C=10, gamma='scale').",
        "Quantum QSVC: 4-qubit ZZFeatureMap with linear entanglement topology.",
        "Fidelity Quantum Kernel: Evaluates statevector overlap inner products.",
        "Stratified train/test splitting for equitable class representation."
    ])

    add_card(s3, 8.8, 1.8, 3.7, 5.0, "3. Interactive Web Dashboard", [
        "Flask REST API backend with asynchronous background training.",
        "Real-time sensor waveform trends and physics-based fault injection.",
        "Dataset analytics: correlation heatmaps, skewness, 5-number stats.",
        "Actionable engineering maintenance checklists and severity rankings."
    ])

    # -------------------------------------------------------------
    # SLIDE 4: Sensor Simulation & Fault Taxonomy
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_background(s4)
    add_header(s4, "Multi-Sensor Simulation & Industrial Fault Taxonomy")

    add_card(s4, 0.8, 1.8, 5.6, 5.0, "3 Simulated Sensor Channels", [
        "Temperature Sensor (PT100 RTD): Nominal 75.0 °C (std 2.0 °C) — tracks thermodynamic balance.",
        "Pressure Sensor (Piezoelectric): Nominal 101.3 kPa (std 1.5 kPa) — monitors pneumatic/hydraulic lines.",
        "Vibration Sensor (Tri-Axial MEMS): Nominal 0.500 mm/s (std 0.05 mm/s) — captures mechanical bearing oscillation.",
        "Signals incorporate realistic baseline Gaussian variance and environmental noise."
    ])

    add_card(s4, 6.8, 1.8, 5.7, 5.0, "5 Categorical Fault States", [
        "Class 0: Normal — Nominal operating baseline across all channels.",
        "Class 1: Drift Fault — Monotonic linear ramp over time (+0.15 °C/s) mimicking calibration loss.",
        "Class 2: Stuck Sensor — Signal variance frozen at constant value; sensor lockup or DAQ failure.",
        "Class 3: High Noise — Excessive variance (3.5x–4.0x standard deviation) representing EMI or looseness.",
        "Class 4: Transient Spike — Extreme impulse excursions (10x variance) representing hydraulic shock."
    ], ACCENT_GREEN)

    # -------------------------------------------------------------
    # SLIDE 5: Sliding-Window Feature Engineering Pipeline
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_background(s5)
    add_header(s5, "Sliding-Window Feature Engineering & Zero-Leakage Preprocessing")

    add_card(s5, 0.8, 1.8, 5.6, 5.0, "15 Extracted Temporal Features", [
        "Window Size W = 50 readings per temporal window.",
        "Non-Overlapping Stride: Ensures no sample leakage across train/test splits.",
        "Mean (mu): Captures steady-state operational level.",
        "Standard Deviation (sigma): Distinguishes stuck (near 0) from noisy (high sigma) states.",
        "Minimum & Maximum: Captures extreme spike anomalies and peak stress.",
        "Trend Slope: Linear regression coefficient isolating slow parameter drifts."
    ])

    add_card(s5, 6.8, 1.8, 5.7, 5.0, "Zero-Leakage Preprocessing Protocol", [
        "Stratified Train/Test Splitting: 75% Training, 25% Held-Out Testing.",
        "StandardScaler: Mean and variance computed exclusively from training data.",
        "Quantum Angle Mapping: Mutual information feature selection selects top 4 features.",
        "Angles scaled to [0, pi] for Pauli Z rotation gate encoding.",
        "Outlier clipping (2.5x IQR) and duplicate removal prevent synthetic artifacts."
    ])

    # -------------------------------------------------------------
    # SLIDE 6: Quantum Computing & QSVC Algorithm
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_background(s6)
    add_header(s6, "Quantum Kernel Method: ZZFeatureMap & QSVC")

    add_card(s6, 0.8, 1.8, 5.6, 5.0, "Quantum Hilbert-Space Embedding", [
        "Encodes classical features x into quantum state |Phi(x)> using 4 qubits.",
        "ZZFeatureMap Circuit: Hadamard gates create superposition across all qubits.",
        "Phase gates R_z(2*x_i) encode first-order individual feature values.",
        "Entangling CNOT-R_z(2*(pi-x_i)*(pi-x_j))-CNOT gates capture pairwise cross-sensor correlations.",
        "Creates non-linear decision boundaries intractable for classical polynomial expansions."
    ], ACCENT_BLUE)

    add_card(s6, 6.8, 1.8, 5.7, 5.0, "Fidelity Quantum Kernel Evaluation", [
        "Measures similarity via quantum state fidelity: K(x_i, x_j) = |<Phi(x_i) | Phi(x_j)>|^2.",
        "Evaluated via statevector simulation on local classical CPU.",
        "Quantum Support Vector Classifier (QSVC): Uses quantum Gram matrix in dual SVM optimization.",
        "Maximizes margin between fault hyperplanes in 2^4 = 16-dimensional quantum state space."
    ], ACCENT_TEAL)

    # -------------------------------------------------------------
    # SLIDE 7: Custom Dataset Lab & Dataset Analytics
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_background(s7)
    add_header(s7, "Custom Dataset Lab & Statistical Analytics")

    add_card(s7, 0.8, 1.8, 5.6, 5.0, "Custom CSV Ingestion Engine", [
        "Supports drag-and-drop file upload for custom industrial CSV datasets.",
        "Automatic column role suggestion: target label, timestamp, and feature channels.",
        "Mode A: Precomputed features with instant cleaning and validation.",
        "Mode B: Raw streaming sensor logs with automated sliding-window extraction.",
        "Dataset Source Switching: Toggle between Synthetic and Custom datasets without loss."
    ])

    add_card(s7, 6.8, 1.8, 5.7, 5.0, "Interactive Dataset Analytics", [
        "Memory footprint, row/column counts, missing values, and duplicate row detection.",
        "Class balance verification with automated imbalance warnings (ratio < 0.25).",
        "Pearson correlation matrix heatmap ($r in [-1.0, 1.0]$) detecting sensor collinearity.",
        "Numerical summary table: 5-number summary (min, Q1, median, Q3, max), mean, std, skewness."
    ])

    # -------------------------------------------------------------
    # SLIDE 8: Empirical Results & Comparative Benchmark
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_background(s8)
    add_header(s8, "Empirical Benchmarking: Classical SVM vs. Quantum QSVC")

    add_card(s8, 0.8, 1.8, 5.6, 5.0, "Empirical Performance Metrics", [
        "Classical SVM (RBF Kernel): ~98.0% Accuracy, 0.98 Macro F1, 0.98 Weighted F1.",
        "Quantum QSVC (ZZFeatureMap): ~87.5% Accuracy, 0.87 Macro F1, 0.88 Weighted F1.",
        "Classical Training Duration: ~0.05 seconds on 750 training samples.",
        "Quantum Simulation Duration: ~24.5 seconds on 80-sample subset (O(N^2) pairwise kernel circuits).",
        "Inference Latency: 0.12 ms (Classical) vs 85.3 ms (Quantum simulator)."
    ], ACCENT_ORANGE)

    add_card(s8, 6.8, 1.8, 5.7, 5.0, "Objective Scientific Findings", [
        "No Artificial Quantum Advantage: Accurately reflects current NISQ simulator performance.",
        "Hilbert Space Separation: The 4-qubit quantum kernel successfully captures fault boundaries without tuning.",
        "Computational Overhead: Statevector simulation on CPU is ~500x slower due to matrix evaluation.",
        "Conclusion: Validates quantum feasibility for industrial fault recognition while highlighting classical efficiency."
    ], ACCENT_GREEN)

    # -------------------------------------------------------------
    # SLIDE 9: Maintenance Guidance & Future Roadmap
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_background(s9)
    add_header(s9, "Prescriptive Maintenance Advisories & Future Roadmap")

    add_card(s9, 0.8, 1.8, 5.6, 5.0, "Automated Maintenance Advisories", [
        "Prescriptive Engineering Checklists: Actionable inspection steps for technicians.",
        "Severity Categorization: NORMAL (green), MEDIUM (amber), HIGH (orange), CRITICAL (red).",
        "Drift Anomaly: 'Inspect physical probe zero-point bias and clean particulate fouling.'",
        "Stuck Sensor: 'Immediate dispatch: verify 4-20mA wiring, transducer seizure, ADC channel.'",
        "Confidence Probability Meter: Calibrated posteriors with engineering disclaimers."
    ])

    add_card(s9, 6.8, 1.8, 5.7, 5.0, "Future Enhancements & Conclusion", [
        "Hardware Deployment: Migrate from Qiskit Aer simulation to physical IBM Quantum superconducting QPUs.",
        "Quantum Error Mitigation: Implement Zero-Noise Extrapolation (ZNE) and readout calibration.",
        "Industrial Edge DAQ: Deploy MQTT streaming pipeline directly to plant SCADA systems.",
        "Project Deliverable: Fully verified (56 unit/integration tests passing), production-ready web application."
    ], ACCENT_TEAL)

    # Save presentation
    prs.save(output_path)
    print(f"[+] Successfully generated PowerPoint deck: {output_path}")

if __name__ == "__main__":
    create_deck()
