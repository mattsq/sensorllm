"""Signal-to-image transforms for sensor time-series."""

from sensorllm.data.transforms.base import BaseTransform
from sensorllm.data.transforms.spectrogram import MelSpectrogramTransform
from sensorllm.data.transforms.cwt import CWTTransform
from sensorllm.data.transforms.recurrence import RecurrencePlotTransform
from sensorllm.data.transforms.raw_image import RawImageTransform

TRANSFORM_REGISTRY: dict[str, type[BaseTransform]] = {
    "spectrogram": MelSpectrogramTransform,
    "cwt": CWTTransform,
    "recurrence": RecurrencePlotTransform,
    "raw_image": RawImageTransform,
}

__all__ = [
    "BaseTransform",
    "MelSpectrogramTransform",
    "CWTTransform",
    "RecurrencePlotTransform",
    "RawImageTransform",
    "TRANSFORM_REGISTRY",
]
