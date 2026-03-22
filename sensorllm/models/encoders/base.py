"""Abstract base class for sensor encoders."""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class SensorEncoder(nn.Module, abc.ABC):
    """Abstract base class for sensor image encoders.

    A sensor encoder takes a batch of sensor images and produces a sequence of
    dense embedding vectors — one per spatial patch or feature map location.
    These embeddings are then processed by a SensorAdapter to produce token
    embeddings for the LLM.

    Subclasses must implement `forward` and expose the `output_dim` property.

    Contract:
        Input:  torch.Tensor of shape (B, C, H, W) — batch of sensor images
        Output: torch.Tensor of shape (B, N, D) — B=batch, N=patches, D=output_dim
    """

    @abc.abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sensor images into patch embeddings.

        Args:
            images: Sensor image tensor of shape (B, C, H, W).

        Returns:
            Embedding tensor of shape (B, N_patches, output_dim).
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the encoder output embeddings."""
        raise NotImplementedError
