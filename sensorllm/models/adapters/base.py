"""Abstract base class for sensor adapter architectures.

All adapters implement the same interface, making them drop-in swappable
without modifying the LLM input construction logic.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class SensorAdapter(nn.Module, abc.ABC):
    """Abstract base class for sensor→LLM adapters.

    A sensor adapter receives dense embeddings from a sensor encoder and
    projects/compresses them into a fixed-length sequence of token embeddings
    in the LLM's embedding space. The fixed-length output is the key contract
    that makes adapters interchangeable.

    Architecture position:
        SensorEncoder → [SensorAdapter] → LLM token sequence

    Required properties:
        n_output_tokens (int): Fixed number of token embeddings produced per
            sensor segment, regardless of input sequence length. The LLM sees
            exactly this many sensor tokens prepended to the text prompt tokens.

    Required methods:
        forward(sensor_embeddings, attention_mask=None) -> token_embeddings

    Example:
        adapter = LinearProjectionAdapter(
            input_dim=768, output_dim=4096, n_output_tokens=32
        )
        # sensor_embs: (B, N_patches, 768) from a ViT encoder
        sensor_embs = torch.randn(2, 196, 768)
        token_embs = adapter(sensor_embs)
        # token_embs: (B, 32, 4096) — ready to prepend to LLM input
        assert token_embs.shape == (2, 32, 4096)
    """

    @abc.abstractmethod
    def forward(
        self,
        sensor_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project sensor embeddings into LLM token space.

        Args:
            sensor_embeddings: Dense embeddings from sensor encoder,
                shape (B, N_patches, encoder_dim).
            attention_mask: Optional binary mask of shape (B, N_patches)
                where 1 = valid token, 0 = padding. May be None.

        Returns:
            Token embeddings of shape (B, n_output_tokens, llm_hidden_dim).
            The output length is always exactly self.n_output_tokens.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_output_tokens(self) -> int:
        """Fixed number of LLM token embeddings produced per sensor input."""
        raise NotImplementedError
