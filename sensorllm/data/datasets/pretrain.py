"""Sensor pretraining dataset for adapter alignment (Stage 1)."""

from __future__ import annotations

from typing import Any

from sensorllm.data.datasets.base import BaseSensorDataset


class SensorPretrainDataset(BaseSensorDataset):
    """Dataset for sensor-LLM adapter alignment pretraining.

    Each sample pairs a sensor image with a short natural language description
    of the signal characteristics. Used in Stage 1 (adapter alignment) to
    teach the adapter to map sensor features into LLM token space.

    Args:
        data_root: Path to the data directory.
        split: One of 'train', 'val'.
        transform: Instantiated signal-to-image transform.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token sequence length.
    """

    def __init__(self, data_root, split: str, transform, tokenizer, max_length: int = 256, **config) -> None:
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._samples: list = []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError("SensorPretrainDataset.__getitem__() not yet implemented")
