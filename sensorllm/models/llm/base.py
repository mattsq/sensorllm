"""Abstract base class for LLM backbone wrappers."""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class LLMBackbone(nn.Module, abc.ABC):
    """Abstract base class for LLM backbone wrappers.

    Wraps a pretrained causal language model and provides a uniform interface
    for embedding lookup, forward pass with sensor token injection, and text
    generation.
    """

    @abc.abstractmethod
    def get_input_embeddings(self) -> nn.Embedding:
        """Return the LLM's token embedding layer."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def hidden_size(self) -> int:
        """LLM hidden dimension (must match adapter output_dim)."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with pre-computed input embeddings.

        Args:
            inputs_embeds: Token embeddings (B, seq_len, hidden_size).
            attention_mask: Binary mask (B, seq_len).
            labels: Token IDs for loss computation; -100 = ignore.

        Returns:
            Tuple of (logits, loss). Loss is None when labels is None.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Generate text conditioned on input embeddings.

        Args:
            inputs_embeds: Conditioned token embeddings (B, seq_len, hidden_size).
            attention_mask: Binary mask (B, seq_len).
            **generation_kwargs: Passed to HuggingFace generate().

        Returns:
            Generated token ID tensor (B, output_len).
        """
        raise NotImplementedError
