"""Temperature sensor reader (EGT, CHT, oil temp, etc.)."""

from __future__ import annotations

from pathlib import Path

from sensorllm.data.sensors.base import BaseSensorReader, SensorReading


class TemperatureSensorReader(BaseSensorReader):
    """Reader for temperature sensor data.

    Handles exhaust gas temperature (EGT), cylinder head temperature (CHT),
    oil temperature, and other thermal sensors.
    """

    def read(self, path: Path) -> SensorReading:
        """Read temperature data from a file.

        Args:
            path: Path to HDF5 or CSV sensor file.

        Returns:
            SensorReading with temperature time-series array.
        """
        raise NotImplementedError(
            f"TemperatureSensorReader.read() not yet implemented for {path}"
        )
