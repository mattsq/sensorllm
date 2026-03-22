"""Aircraft sensor Q&A dataset for instruction fine-tuning (Stage 2)."""

from __future__ import annotations

from typing import Any

from sensorllm.data.datasets.base import BaseSensorDataset


class AircraftSensorQADataset(BaseSensorDataset):
    """Dataset for aircraft sensor Q&A tasks.

    Each sample pairs a sensor time-series window with a question-answer pair
    about anomaly detection, fault description, or operational state narration.
    Used in Stage 2 (instruction fine-tuning).

    Args:
        data_root: Path to the data directory.
        split: One of 'train', 'val', 'test'.
        transform: Instantiated signal-to-image transform.
        tokenizer: HuggingFace tokenizer for the LLM backbone.
        max_length: Maximum token sequence length.
        config: Full dataset config dict.
    """

    def __init__(self, data_root, split: str, transform, tokenizer, max_length: int = 512, **config) -> None:
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._samples: list = []  # populated by _load_index()
        # self._load_index()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError("AircraftSensorQADataset.__getitem__() not yet implemented")
