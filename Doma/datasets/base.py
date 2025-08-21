import numpy as np
from abc import ABC, abstractmethod

class TimeSeriesDataset(ABC):
    def __init__(self, length: int, seed: int = 42):
        self.length = length
        self.seed = seed
        self.series = None

    @abstractmethod
    def generate(self):
        """Generate or load dataset into self.series."""
        pass

    def get_series(self) -> np.ndarray:
        if self.series is None:
            raise ValueError("Dataset not generated yet. Call generate().")
        return self.series

    def info(self):
        return {
            "length": self.length,
            "seed": self.seed,
            "series_generated": self.series is not None
        }
