"""Abstract base class for sensor datasets."""

from __future__ import annotations

import abc
from typing import Any

import torch
from torch.utils.data import Dataset


class BaseSensorDataset(Dataset, abc.ABC):
    """Abstract base class for sensor-LLM datasets.

    All datasets must return dicts with the following keys so the trainer
    sees a uniform interface regardless of sensor type or task:

        sensor_image   torch.Tensor  (C, H, W) float32 — sensor image
        input_ids      torch.Tensor  (seq_len,) long    — tokenized prompt + answer
        attention_mask torch.Tensor  (seq_len,) long    — 1 for real tokens
        labels         torch.Tensor  (seq_len,) long    — -100 for prompt (masked from loss)
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single training sample.

        Returns:
            Dict with keys: sensor_image, input_ids, attention_mask, labels.
        """
        raise NotImplementedError
