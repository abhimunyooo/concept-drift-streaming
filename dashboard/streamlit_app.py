import sys
import os

# Add src folder to Python path
sys.path.append(os.path.abspath(".."))

import streamlit as st
import matplotlib.pyplot as plt

from src.stream import DataStream
from src.model import OnlineModel
from src.drift import DriftDetector
from src.retrain import Retrainer
from src.metrics import MetricsTracker
from src.config import WINDOW_SIZE

from river import drift


# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="Concept Drift Streaming Demo", layout="wide")

st.title("📈 Real-Time Concept Drift Detection Demo")
st.markdown("Simulated streaming with ADWIN drift detection and model reset.")


# ===============================
# Sidebar Controls
# ===============================

st.sidebar.header("Configuration")

delta = st.sidebar.slider(
    "ADWIN Delta (Sensitivity)",
    min_value=0.0001,
    max_value=0.01,
    value=0.0005,
    step=0.0001
)

run_button = st.sidebar.button("Run Streaming Pipeline")


# ===============================
# Run Pipeline
# ===============================

if run_button:

    st.write("Running streaming pipeline...")

    data_stream = DataStream()
    stream = data_stream.get_stream()

    model = OnlineModel()
    drift_detector = drift.ADWIN(delta=delta)
    metrics = MetricsTracker()

    drift_points = []
    reset_count = 0

    for i, (x, y) in enumerate(stream):

        y_pred = model.predict(x)
        metrics.update(y, y_pred)

        error = 0 if y_pred is None else int(y_pred != y)

        drift_detector.update(error)

        if drift_detector.drift_detected:
            drift_points.append(i)
            model.reset()
            drift_detector = drift.ADWIN(delta=delta)
            reset_count += 1

        model.learn(x, y)

    # ===============================
    # Display Metrics
    # ===============================

    final_accuracy = metrics.get_final_accuracy()
    worst_rolling = metrics.get_worst_rolling_accuracy()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Final Accuracy", f"{final_accuracy:.4f}")
    col2.metric("Worst Rolling Accuracy", f"{worst_rolling:.4f}")
    col3.metric("Drift Detections", len(drift_points))
    col4.metric("Model Resets", reset_count)

    # ===============================
    # Plot
    # ===============================

    rolling_acc = metrics.get_rolling_accuracy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rolling_acc)

    for dp in drift_points:
        ax.axvline(dp, linestyle="--", alpha=0.4)

    ax.set_title("Rolling Accuracy with Drift Detection")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Accuracy")

    st.pyplot(fig)
