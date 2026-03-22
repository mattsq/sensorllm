"""Pressure sensor reader (manifold pressure, hydraulic, fuel pressure, etc.)."""

from __future__ import annotations

from pathlib import Path

from sensorllm.data.sensors.base import BaseSensorReader, SensorReading


class PressureSensorReader(BaseSensorReader):
    """Reader for pressure sensor data.

    Handles manifold absolute pressure (MAP), hydraulic pressure, fuel pressure,
    cabin pressure differential, and other pneumatic sensors.
    """

    def read(self, path: Path) -> SensorReading:
        """Read pressure data from a file.

        Args:
            path: Path to HDF5 or CSV sensor file.

        Returns:
            SensorReading with pressure time-series array.
        """
        raise NotImplementedError(
            f"PressureSensorReader.read() not yet implemented for {path}"
        )
