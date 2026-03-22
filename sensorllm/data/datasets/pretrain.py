"""Sensor pretraining dataset for adapter alignment (Stage 1)."""

from __future__ import annotations

from typing import Any

from sensorllm.data.datasets.base import BaseSensorDataset


class SensorPretrainDataset(BaseSensorDataset):
    """Dataset for sensor-LLM adapter alignment pretraining.

    Each sample pairs a windowed sensor time-series with a short natural language
    description of the signal characteristics. Used in Stage 1 (adapter alignment)
    to teach the adapter and encoder to map sensor features into LLM token space.

    The raw sensor window is returned as-is (float32, shape (C, L)).

    Args:
        data_root: Path to the data directory.
        split: One of 'train', 'val'.
        tokenizer: HuggingFace tokenizer.
        window_size: Number of samples per window.
        n_channels: Number of sensor channels.
        max_length: Maximum token sequence length.
    """

    def __init__(
        self,
        data_root,
        split: str,
        tokenizer,
        window_size: int = 4096,
        n_channels: int = 1,
        max_length: int = 256,
        **config,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.n_channels = n_channels
        self.max_length = max_length
        self._samples: list = []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError("SensorPretrainDataset.__getitem__() not yet implemented")
