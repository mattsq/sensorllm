"""Data pipeline: sensor readers, windowing, and PyTorch datasets."""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy-load torch-dependent sub-registries on first access."""
    if name == "SENSOR_REGISTRY":
        from sensorllm.data.sensors import SENSOR_REGISTRY  # noqa: PLC0415
        return SENSOR_REGISTRY
    if name == "DATASET_REGISTRY":
        from sensorllm.data.datasets import DATASET_REGISTRY  # noqa: PLC0415
        return DATASET_REGISTRY
    raise AttributeError(f"module 'sensorllm.data' has no attribute {name!r}")


__all__ = ["SENSOR_REGISTRY", "DATASET_REGISTRY"]
