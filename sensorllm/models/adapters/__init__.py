"""Sensor adapter architectures: bridge between sensor encoder and LLM token space."""

from sensorllm.models.adapters.base import SensorAdapter
from sensorllm.models.adapters.linear_projection import LinearProjectionAdapter
from sensorllm.models.adapters.qformer import QFormerAdapter
from sensorllm.models.adapters.perceiver import PerceiverResamplerAdapter
from sensorllm.models.adapters.mlp_mixer import MLPMixerAdapter

ADAPTER_REGISTRY: dict[str, type[SensorAdapter]] = {
    "linear_projection": LinearProjectionAdapter,
    "qformer": QFormerAdapter,
    "perceiver": PerceiverResamplerAdapter,
    "mlp_mixer": MLPMixerAdapter,
}

__all__ = [
    "SensorAdapter",
    "LinearProjectionAdapter",
    "QFormerAdapter",
    "PerceiverResamplerAdapter",
    "MLPMixerAdapter",
    "ADAPTER_REGISTRY",
]
