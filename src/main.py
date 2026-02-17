import matplotlib.pyplot as plt

from stream import DataStream
from model import OnlineModel
from drift import DriftDetector
from retrain import Retrainer
from metrics import MetricsTracker
from config import VERBOSE


def main():

    # ===============================
    # Initialize Components
    # ===============================
    data_stream = DataStream()
    stream = data_stream.get_stream()

    model = OnlineModel()
    drift_detector = DriftDetector()
    metrics = MetricsTracker()
    retrainer = Retrainer(model, drift_detector)

    drift_points = []

    if VERBOSE:
        print("Starting streaming pipeline...")
        print(f"Total samples: {data_stream.get_length()}")

    # ===============================
    # Streaming Loop
    # ===============================
    for i, (x, y) in enumerate(stream):

        # Predict
        y_pred = model.predict(x)

        # Update metrics
        metrics.update(y, y_pred)

        # Compute error for drift detection
        error = 0 if y_pred is None else int(y_pred != y)

        # Update drift detector
        drift_detected = drift_detector.update(error)

        if drift_detected:
            drift_points.append(i)
            retrainer.handle_drift()

            if VERBOSE:
                print(f"Drift detected at index {i}")

        # Learn
        model.learn(x, y)

    # ===============================
    # Final Statistics
    # ===============================
    final_accuracy = metrics.get_final_accuracy()
    worst_rolling = metrics.get_worst_rolling_accuracy()

    print("\n===== STREAMING COMPLETE =====")
    print(f"Final Cumulative Accuracy: {final_accuracy:.4f}")
    print(f"Worst Rolling Accuracy:     {worst_rolling:.4f}")
    print(f"Total Drift Detections:     {len(drift_points)}")
    print(f"Total Model Resets:         {retrainer.reset_count}")

    # ===============================
    # Plot Rolling Accuracy
    # ===============================
    rolling_acc = metrics.get_rolling_accuracy()

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_acc, label="Rolling Accuracy")

    for dp in drift_points:
        plt.axvline(dp, color="red", linestyle="--", alpha=0.4)

    plt.title("Streaming Performance with Drift Detection")
    plt.xlabel("Sample Index")
    plt.ylabel("Rolling Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
