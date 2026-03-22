"""Sensor modality readers."""

from sensorllm.data.sensors.base import BaseSensorReader, SensorReading
from sensorllm.data.sensors.imu import IMUSensorReader
from sensorllm.data.sensors.vibration import VibrationSensorReader
from sensorllm.data.sensors.temperature import TemperatureSensorReader
from sensorllm.data.sensors.pressure import PressureSensorReader

SENSOR_REGISTRY: dict[str, type[BaseSensorReader]] = {
    "imu": IMUSensorReader,
    "vibration": VibrationSensorReader,
    "temperature": TemperatureSensorReader,
    "pressure": PressureSensorReader,
}

__all__ = [
    "BaseSensorReader",
    "SensorReading",
    "IMUSensorReader",
    "VibrationSensorReader",
    "TemperatureSensorReader",
    "PressureSensorReader",
    "SENSOR_REGISTRY",
]
