"""SensorLLM: Image adapter approaches for aircraft sensor-LLM fusion."""

from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy-load heavy model/utility symbols so the package can be imported
    without PyTorch installed (e.g. when only the data utilities are needed)."""
    if name == "SensorLLMModel":
        from sensorllm.models.sensorllm_model import SensorLLMModel  # noqa: PLC0415
        return SensorLLMModel
    if name == "load_config":
        from sensorllm.utils.config import load_config  # noqa: PLC0415
        return load_config
    if name == "get_logger":
        from sensorllm.utils.logging import get_logger  # noqa: PLC0415
        return get_logger
    raise AttributeError(f"module 'sensorllm' has no attribute {name!r}")


__all__ = ["SensorLLMModel", "load_config", "get_logger", "__version__"]
