"""IMU (accelerometer + gyroscope) sensor reader."""

from __future__ import annotations

from pathlib import Path

from sensorllm.data.sensors.base import BaseSensorReader, SensorReading


class IMUSensorReader(BaseSensorReader):
    """Reader for IMU sensor data (accelerometer and/or gyroscope).

    Supports HDF5 and CSV file formats. Expects 3-axis accelerometer and/or
    3-axis gyroscope channels.

    Args:
        channels: List of channel names to load. Defaults to all available channels.
    """

    def __init__(self, channels: list[str] | None = None) -> None:
        self.channels = channels

    def read(self, path: Path) -> SensorReading:
        """Read IMU data from a file.

        Args:
            path: Path to HDF5 or CSV sensor file.

        Returns:
            SensorReading with shape (n_samples, n_channels) signal array.
        """
        raise NotImplementedError(f"IMUSensorReader.read() not yet implemented for {path}")
