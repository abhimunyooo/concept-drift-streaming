import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(".."))

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from src.stream import DataStream
from src.model import OnlineModel
from src.metrics import MetricsTracker
from river import drift


# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(page_title="Concept Drift Demonstration", layout="wide")

st.title("Concept Drift Demonstration in Streaming ML")

st.markdown(
"""
This demo shows how model performance degrades when the data distribution changes (concept drift),
and how statistical drift detection with ADWIN enables faster recovery.

Rolling accuracy drops indicate distribution shifts.
Vertical lines indicate detected drift events.
"""
)


# ===============================
# SIDEBAR CONTROLS
# ===============================

st.sidebar.header("Experiment Settings")

mode = st.sidebar.radio(
    "Execution Mode",
    ["Baseline (No Adaptation)", 
     "Adaptive (ADWIN Reset)", 
     "Comparison (Baseline vs Adaptive)"]
)

delta = st.sidebar.slider(
    "ADWIN Delta (Sensitivity)",
    min_value=0.0001,
    max_value=0.01,
    value=0.0005,
    step=0.0001
)

run_button = st.sidebar.button("Run Experiment")


# ===============================
# HELPER FUNCTIONS
# ===============================

def run_baseline():
    data_stream = DataStream()
    stream = data_stream.get_stream()

    model = OnlineModel()
    metrics = MetricsTracker()

    for x, y in stream:
        y_pred = model.predict(x)
        metrics.update(y, y_pred)
        model.learn(x, y)

    return metrics, []


def run_adaptive(delta):
    data_stream = DataStream()
    stream = data_stream.get_stream()

    model = OnlineModel()
    metrics = MetricsTracker()
    detector = drift.ADWIN(delta=delta)

    drift_points = []
    reset_count = 0

    for i, (x, y) in enumerate(stream):

        y_pred = model.predict(x)
        metrics.update(y, y_pred)

        error = 0 if y_pred is None else int(y_pred != y)

        detector.update(error)

        if detector.drift_detected:
            drift_points.append(i)
            model.reset()
            detector = drift.ADWIN(delta=delta)
            reset_count += 1

        model.learn(x, y)

    return metrics, drift_points


# ===============================
# EXECUTION
# ===============================

if run_button:

    st.write("Running experiment...")

    if mode == "Baseline (No Adaptation)":
        metrics, drift_points = run_baseline()

        rolling = metrics.get_rolling_accuracy()

        st.subheader("Baseline Performance (No Drift Handling)")
        st.markdown(
        """
        The model continuously learns but does not reset when drift occurs.
        Notice prolonged accuracy degradation after distribution shifts.
        """
        )

        col1, col2 = st.columns(2)
        col1.metric("Final Accuracy", f"{metrics.get_final_accuracy():.4f}")
        col2.metric("Worst Rolling Accuracy", f"{metrics.get_worst_rolling_accuracy():.4f}")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(rolling)
        ax.set_title("Rolling Accuracy - Baseline")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)


    elif mode == "Adaptive (ADWIN Reset)":
        metrics, drift_points = run_adaptive(delta)

        rolling = metrics.get_rolling_accuracy()

        st.subheader("Adaptive Performance (Drift Detection Enabled)")
        st.markdown(
        """
        ADWIN monitors prediction error.
        When a statistical shift is detected, the model is reset.
        Notice faster recovery after each drift.
        """
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Accuracy", f"{metrics.get_final_accuracy():.4f}")
        col2.metric("Worst Rolling Accuracy", f"{metrics.get_worst_rolling_accuracy():.4f}")
        col3.metric("Drift Detections", len(drift_points))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(rolling)

        for dp in drift_points:
            ax.axvline(dp, linestyle="--", alpha=0.4)

        ax.set_title("Rolling Accuracy - Adaptive")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)


    else:  # Comparison Mode

        st.subheader("Baseline vs Adaptive Comparison")

        baseline_metrics, _ = run_baseline()
        adaptive_metrics, drift_points = run_adaptive(delta)

        rolling_base = baseline_metrics.get_rolling_accuracy()
        rolling_adapt = adaptive_metrics.get_rolling_accuracy()

        col1, col2 = st.columns(2)
        col1.metric("Baseline Final Accuracy", f"{baseline_metrics.get_final_accuracy():.4f}")
        col2.metric("Adaptive Final Accuracy", f"{adaptive_metrics.get_final_accuracy():.4f}")

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(rolling_base, label="Baseline")
        ax.plot(rolling_adapt, label="Adaptive")

        for dp in drift_points:
            ax.axvline(dp, linestyle="--", alpha=0.3)

        ax.set_title("Rolling Accuracy Comparison")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Accuracy")
        ax.legend()

        st.pyplot(fig)

        st.markdown(
        """
        Interpretation:

        • Large drops indicate distribution shifts.  
        • Baseline recovers slowly because it adapts incrementally.  
        • Adaptive mode resets the model, shortening recovery time.  
        • Drift markers show where statistical change was detected.  
        """
        )
