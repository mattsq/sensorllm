"""Abstract base class for sensor data readers."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import NamedTuple

import numpy as np


class SensorReading(NamedTuple):
    """Standardized output from any sensor reader.

    Attributes:
        signal: Raw signal array of shape (n_samples, n_channels).
        sample_rate: Sampling frequency in Hz.
        metadata: Arbitrary key-value metadata (flight ID, sensor ID, units, etc.).
    """

    signal: np.ndarray
    sample_rate: float
    metadata: dict


class BaseSensorReader(abc.ABC):
    """Abstract base class for sensor data readers.

    All sensor readers must implement the `read` method, which takes a path to a
    raw sensor file and returns a `SensorReading` namedtuple.

    Example:
        reader = IMUSensorReader()
        reading = reader.read(Path("data/raw/flight_001.h5"))
        print(reading.signal.shape)   # (n_samples, n_channels)
        print(reading.sample_rate)    # e.g., 400.0
    """

    @abc.abstractmethod
    def read(self, path: Path) -> SensorReading:
        """Read a sensor file and return a standardized SensorReading.

        Args:
            path: Path to the raw sensor data file.

        Returns:
            SensorReading with signal array, sample rate, and metadata dict.
        """
        raise NotImplementedError
