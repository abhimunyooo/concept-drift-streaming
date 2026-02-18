import pandas as pd
from river import stream as river_stream

from .config import DATA_PATH


class DataStream:
    """
    DataStream handles:
    - Loading dataset
    - Converting it into a streaming generator
    """

    def __init__(self):
        self.data = pd.read_csv(DATA_PATH)
        self.X = self.data.drop(columns=["label"])
        self.y = self.data["label"]

    def get_stream(self):
        """
        Returns a generator yielding (xi, yi)
        compatible with river models.
        """
        return river_stream.iter_pandas(self.X, self.y)

    def get_length(self):
        """
        Returns total number of samples.
        """
        return len(self.data)
