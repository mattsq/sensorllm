"""PyTorch Dataset classes for sensor-LLM training."""

from sensorllm.data.datasets.base import BaseSensorDataset
from sensorllm.data.datasets.aircraft_qa import AircraftSensorQADataset
from sensorllm.data.datasets.pretrain import SensorPretrainDataset

DATASET_REGISTRY: dict[str, type[BaseSensorDataset]] = {
    "aircraft_qa": AircraftSensorQADataset,
    "pretrain": SensorPretrainDataset,
}

__all__ = [
    "BaseSensorDataset",
    "AircraftSensorQADataset",
    "SensorPretrainDataset",
    "DATASET_REGISTRY",
]
