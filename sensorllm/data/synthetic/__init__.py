"""Synthetic sensor data generation for end-to-end pipeline smoke testing."""

from sensorllm.data.synthetic.sensor_generator import (
    EventType,
    SensorType,
    SyntheticSensorConfig,
    generate_imu_signal,
    generate_pressure_signal,
    generate_temperature_signal,
    generate_vibration_signal,
)
from sensorllm.data.synthetic.annotation_generator import AnnotationGenerator
from sensorllm.data.synthetic.dataset_builder import SyntheticDatasetBuilder

__all__ = [
    "EventType",
    "SensorType",
    "SyntheticSensorConfig",
    "generate_vibration_signal",
    "generate_imu_signal",
    "generate_temperature_signal",
    "generate_pressure_signal",
    "AnnotationGenerator",
    "SyntheticDatasetBuilder",
]
