"""Abstract base class for time-series sensor encoders."""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class SensorEncoder(nn.Module, abc.ABC):
    """Abstract base class for sensor time-series encoders.

    A sensor encoder takes a batch of windowed sensor signals and produces a
    sequence of dense latent embedding vectors — one per temporal patch or
    feature map position. These embeddings are then processed by a SensorAdapter
    to produce token embeddings for the LLM.

    No image conversion is involved. Sensor data is encoded directly in the
    time-series domain.

    Subclasses must implement `forward` and expose the `output_dim` property.

    Contract:
        Input:  torch.Tensor of shape (B, C, L)
                  B = batch size
                  C = number of sensor channels (e.g., 1 for single-axis vibration,
                      3 for 3-axis IMU accelerometer)
                  L = window length in samples
        Output: torch.Tensor of shape (B, N, D)
                  N = number of temporal patches / latent tokens
                  D = output_dim (encoder embedding dimensionality)
    """

    @abc.abstractmethod
    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sensor signals into temporal patch embeddings.

        Args:
            signals: Windowed sensor signal tensor of shape (B, C, L).

        Returns:
            Embedding tensor of shape (B, N_patches, output_dim).
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the encoder output embeddings."""
        raise NotImplementedError
