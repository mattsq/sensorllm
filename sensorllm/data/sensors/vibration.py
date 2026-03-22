"""Vibration sensor reader (accelerometers on engine/airframe)."""

from __future__ import annotations

from pathlib import Path

from sensorllm.data.sensors.base import BaseSensorReader, SensorReading


class VibrationSensorReader(BaseSensorReader):
    """Reader for vibration sensor data.

    Typically high-frequency (1–20 kHz) single or multi-axis accelerometer data
    mounted on engine components, gearboxes, or airframe structure.

    Args:
        sample_rate_override: If set, overrides the sample rate from file metadata.
    """

    def __init__(self, sample_rate_override: float | None = None) -> None:
        self.sample_rate_override = sample_rate_override

    def read(self, path: Path) -> SensorReading:
        """Read vibration data from a file.

        Args:
            path: Path to HDF5 or CSV sensor file.

        Returns:
            SensorReading with high-frequency signal array.
        """
        raise NotImplementedError(f"VibrationSensorReader.read() not yet implemented for {path}")
