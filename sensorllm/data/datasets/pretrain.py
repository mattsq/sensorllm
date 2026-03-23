"""Sensor pretraining dataset for adapter alignment (Stage 1)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from sensorllm.data.datasets.base import BaseSensorDataset

logger = logging.getLogger(__name__)


class SensorPretrainDataset(BaseSensorDataset):
    """Dataset for sensor-LLM adapter alignment pretraining.

    Each sample pairs a windowed sensor time-series with a short natural language
    description of the signal characteristics. Used in Stage 1 (adapter alignment)
    to teach the adapter and encoder to map sensor features into LLM token space.

    The raw sensor window is returned as-is (float32, shape (C, L)).

    Args:
        data_root: Path to the data directory.
        split: One of 'train', 'val', 'test'.
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
        sensors: list[str] | None = None,
        **config,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.n_channels = n_channels
        self.max_length = max_length
        self.sensors = sensors
        self._samples: list[dict] = []
        self._load_index()

    def _load_index(self) -> None:
        """Read the JSONL split file and populate self._samples.

        If ``self.sensors`` is set, only samples whose ``sensor`` field matches
        one of the specified sensor types are loaded.
        """
        index_path = self.data_root / "splits" / f"synthetic_{self.split}.jsonl"
        if not index_path.exists():
            logger.warning("Index file not found: %s", index_path)
            return
        sensor_filter = set(self.sensors) if self.sensors else None
        with index_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if sensor_filter and record.get("sensor") not in sensor_filter:
                        continue
                    self._samples.append(record)
        logger.info("Loaded %d samples from %s", len(self._samples), index_path)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self._samples[idx]

        # Read sensor signal from HDF5
        h5_path = self.data_root / record["path"]
        with h5py.File(h5_path, "r") as f:
            signal = f["signal"][:].astype(np.float32)  # (n_samples, n_channels)

        # Transpose to (C, L) for PyTorch convention
        signal = signal.T  # (n_channels, n_samples)

        # Crop or pad to window_size
        L = signal.shape[1]
        if L >= self.window_size:
            signal = signal[:, : self.window_size]
        else:
            pad = np.zeros((signal.shape[0], self.window_size - L), dtype=np.float32)
            signal = np.concatenate([signal, pad], axis=1)

        sensor_signal = torch.from_numpy(signal)

        # Build text: instruction + response
        instruction = (
            "You are analyzing aircraft sensor data. "
            "Describe the sensor reading in one or two sentences."
        )
        response = record.get("description", "")
        full_text = instruction + "\n" + response

        # Tokenize instruction (for label masking) and full text
        instruction_enc = self.tokenizer(
            instruction + "\n",
            add_special_tokens=False,
        )
        n_instruction_tokens = len(instruction_enc["input_ids"])

        full_enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Build labels: -100 for instruction tokens, keep response tokens
        labels = input_ids.clone()
        labels[:n_instruction_tokens] = -100
        # Also mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            "sensor_signal": sensor_signal,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
