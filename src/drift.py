from river import drift
from .config import ADWIN_DELTA


class DriftDetector:
    """
    Wrapper around River's ADWIN drift detector.

    Responsibilities:
    - Maintain ADWIN instance
    - Update with error values
    - Signal when drift is detected
    """

    def __init__(self):
        self._initialize_detector()

    def _initialize_detector(self):
        """
        Initialize ADWIN with configured delta.
        """
        self.detector = drift.ADWIN(delta=ADWIN_DELTA)

    def update(self, error):
        """
        Update ADWIN with prediction error.
        Returns True if drift is detected.
        """
        self.detector.update(error)
        return self.detector.drift_detected

    def reset(self):
        """
        Reset the drift detector.
        """
        self._initialize_detector()
