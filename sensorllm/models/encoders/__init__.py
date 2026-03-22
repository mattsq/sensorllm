"""Sensor image encoders (vision backbone side of the adapter pipeline)."""

from sensorllm.models.encoders.base import SensorEncoder
from sensorllm.models.encoders.vit_encoder import ViTSensorEncoder
from sensorllm.models.encoders.resnet_encoder import ResNetSensorEncoder
from sensorllm.models.encoders.cnn1d_encoder import CNN1DSensorEncoder

ENCODER_REGISTRY: dict[str, type[SensorEncoder]] = {
    "vit_b16": ViTSensorEncoder,
    "resnet50": ResNetSensorEncoder,
    "cnn1d": CNN1DSensorEncoder,
}

__all__ = [
    "SensorEncoder",
    "ViTSensorEncoder",
    "ResNetSensorEncoder",
    "CNN1DSensorEncoder",
    "ENCODER_REGISTRY",
]
