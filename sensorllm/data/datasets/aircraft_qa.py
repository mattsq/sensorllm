"""Aircraft sensor Q&A dataset for instruction fine-tuning (Stage 2)."""

from __future__ import annotations

from typing import Any

from sensorllm.data.datasets.base import BaseSensorDataset


class AircraftSensorQADataset(BaseSensorDataset):
    """Dataset for aircraft sensor Q&A tasks.

    Each sample pairs a windowed sensor time-series with a question-answer pair
    about anomaly detection, fault description, or operational state narration.
    Used in Stage 2 (instruction fine-tuning).

    The raw sensor window is returned as-is (float32, shape (C, L)) — no image
    transform. The SensorEncoder in the model processes it directly.

    Args:
        data_root: Path to the data directory.
        split: One of 'train', 'val', 'test'.
        tokenizer: HuggingFace tokenizer for the LLM backbone.
        window_size: Number of samples per window (= L dimension of sensor_signal).
        n_channels: Number of sensor channels (= C dimension of sensor_signal).
        max_length: Maximum token sequence length.
        config: Additional config kwargs.
    """

    def __init__(
        self,
        data_root,
        split: str,
        tokenizer,
        window_size: int = 4096,
        n_channels: int = 1,
        max_length: int = 512,
        **config,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.n_channels = n_channels
        self.max_length = max_length
        self._samples: list = []  # populated by _load_index()
        # self._load_index()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError("AircraftSensorQADataset.__getitem__() not yet implemented")
