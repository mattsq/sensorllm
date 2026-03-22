"""Time-series sensor encoders."""

from sensorllm.models.encoders.base import SensorEncoder
from sensorllm.models.encoders.cnn1d_encoder import CNN1DSensorEncoder
from sensorllm.models.encoders.transformer_encoder import TransformerSensorEncoder
from sensorllm.models.encoders.patchtst_encoder import PatchTSTSensorEncoder

ENCODER_REGISTRY: dict[str, type[SensorEncoder]] = {
    "cnn1d": CNN1DSensorEncoder,
    "transformer": TransformerSensorEncoder,
    "patchtst": PatchTSTSensorEncoder,
}

__all__ = [
    "SensorEncoder",
    "CNN1DSensorEncoder",
    "TransformerSensorEncoder",
    "PatchTSTSensorEncoder",
    "ENCODER_REGISTRY",
]
