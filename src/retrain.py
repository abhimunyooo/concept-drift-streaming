from config import RETRAIN_MODE


class Retrainer:
    """
    Handles retraining logic when drift is detected.

    Currently supports:
    - Hard reset

    Future extensions:
    - Sliding window retraining
    - Partial reset
    """

    def __init__(self, model, drift_detector):
        self.model = model
        self.drift_detector = drift_detector
        self.reset_count = 0

    def handle_drift(self):
        """
        Executes retraining strategy when drift is detected.
        """
        if RETRAIN_MODE == "reset":
            self.model.reset()
            self.drift_detector.reset()
            self.reset_count += 1
