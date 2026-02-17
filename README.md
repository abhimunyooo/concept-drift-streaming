# Real-Time Concept Drift Detection in Streaming Data

## Overview

This project implements a real-time streaming machine learning pipeline capable of detecting and adapting to **concept drift** in non-stationary environments.

The system simulates a continuous transaction stream using a synthetic Rolling Torus dataset containing multiple abrupt distribution shifts. It uses:

* Online learning (Hoeffding Tree)
* ADWIN drift detection
* Automatic retraining via model reset
* Rolling performance evaluation
* Interactive Streamlit dashboard

The goal is to demonstrate how streaming ML systems degrade under drift and how adaptive retraining significantly improves recovery and stability.

---

## Problem Statement

In streaming environments such as fraud detection, sensor monitoring, or financial transactions, the underlying data distribution can change over time. This phenomenon is known as **concept drift**.

Without drift handling:

* Model performance degrades
* Recovery is slow
* Prediction reliability drops
* Business risk increases

This project builds a fully modular system to:

1. Simulate streaming data
2. Detect drift in real time
3. Trigger retraining automatically
4. Quantitatively evaluate recovery improvement

---

## Dataset

The project uses:

`nonlinear_sudden_rollingtorus_noise_and_redunce.csv`

This dataset contains:

* 100,000 samples
* 5 numerical features
* Binary classification labels
* Multiple abrupt drift points

The Rolling Torus generator creates nonlinear class boundaries that shift over time, simulating real-world non-stationary behavior.

---

## System Architecture

The pipeline is fully modular:

* `DataStream` — Handles streaming simulation
* `OnlineModel` — Wraps Hoeffding Tree classifier
* `DriftDetector` — Implements ADWIN
* `Retrainer` — Executes model reset on drift
* `MetricsTracker` — Tracks rolling accuracy and performance

During streaming:

1. Predict
2. Update metrics
3. Feed prediction error into ADWIN
4. Detect drift
5. Reset model
6. Continue learning

This mimics production-grade event-driven ML systems.

---

## Drift Detection

The system uses ADWIN (Adaptive Windowing), a statistically grounded change detection algorithm.

ADWIN monitors prediction error and detects distribution shifts when the error rate changes significantly.

Configurable sensitivity via:

```
ADWIN_DELTA
```

Lower delta → more conservative
Higher delta → more sensitive

---

## Experimental Results

Three configurations were evaluated:

1. No retraining
2. ADWIN reset (default sensitivity)
3. Tuned ADWIN reset (delta = 0.0005)

Final production results using tuned ADWIN:

* Final Cumulative Accuracy: **0.9436**
* Worst Rolling Accuracy: **0.8695**
* Drift Detections: 25
* Model Resets: 25

The tuned adaptive system significantly reduced catastrophic performance collapse compared to the non-adaptive baseline.

---

## Performance Visualization

The following plot shows rolling accuracy over time with detected drift points marked by vertical lines.

![Streaming Performance with Drift Detection](results/production_rolling_accuracy.png)

Observations:

* Clear accuracy drops at drift points
* Rapid recovery after reset
* No prolonged catastrophic degradation
* Stable performance in later segments

This demonstrates effective detection and adaptation.

---

## Streamlit Dashboard

An interactive dashboard allows real-time experimentation:

* Adjust ADWIN sensitivity
* Observe drift detection frequency
* Monitor rolling accuracy
* Inspect reset count

Run:

```
cd dashboard
streamlit run streamlit_app.py
```

This transforms the project into an interactive streaming ML demonstration.

---

## Key Takeaways

* Concept drift causes measurable performance degradation.
* Online learning alone adapts slowly.
* Statistical drift detection enables rapid recovery.
* Sensitivity tuning balances stability and responsiveness.
* Modular design enables production-ready extensibility.

---

## Possible Extensions

* Replace reset with sliding-window retraining
* Add Adaptive Random Forest
* Compare multiple drift detectors (Page-Hinkley, DDM)
* Add class-wise precision/recall tracking
* Integrate real streaming source (Kafka)

---

## Technologies Used

* Python
* River (online machine learning)
* ADWIN drift detection
* Pandas / NumPy
* Matplotlib
* Streamlit

---

## Conclusion

This project demonstrates a full lifecycle streaming ML system with automated drift detection and retraining. It bridges research-level experimentation with production-grade modular architecture.

It serves as a foundation for building robust, adaptive machine learning systems in non-stationary environments.
