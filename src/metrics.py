import pandas as pd
from config import WINDOW_SIZE


class MetricsTracker:
    """
    Tracks streaming performance metrics.

    Responsibilities:
    - Store prediction correctness
    - Compute rolling accuracy
    - Compute summary statistics
    """

    def __init__(self):
        self.correct_predictions = []

    def update(self, y_true, y_pred):
        """
        Update metrics with new prediction result.
        """
        if y_pred is None:
            self.correct_predictions.append(0)
        else:
            self.correct_predictions.append(int(y_true == y_pred))

    def get_rolling_accuracy(self):
        """
        Returns rolling accuracy as pandas Series.
        """
        series = pd.Series(self.correct_predictions)
        return series.rolling(WINDOW_SIZE).mean()

    def get_final_accuracy(self):
        """
        Returns final cumulative accuracy.
        """
        if len(self.correct_predictions) == 0:
            return 0.0
        return sum(self.correct_predictions) / len(self.correct_predictions)

    def get_worst_rolling_accuracy(self):
        """
        Returns minimum rolling accuracy.
        """
        rolling = self.get_rolling_accuracy()
        return rolling.min()
