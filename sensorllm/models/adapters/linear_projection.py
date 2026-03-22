"""Linear projection adapter — LLaVA-1 style two-layer MLP bridge."""

from __future__ import annotations

import torch
import torch.nn as nn

from sensorllm.models.adapters.base import SensorAdapter


class LinearProjectionAdapter(SensorAdapter):
    """Two-layer MLP that projects sensor embeddings into LLM token space.

    The simplest adapter approach (LLaVA-1 style). First flattens the spatial
    sequence to a fixed number of tokens via linear interpolation, then applies
    a two-layer MLP projection to match the LLM hidden dimension.

    Fast to train, good baseline. Limited capacity for complex feature transformation.

    Args:
        input_dim: Dimensionality of sensor encoder output embeddings.
        output_dim: LLM hidden dimension (target token embedding size).
        n_tokens: Number of output tokens to produce.
        hidden_dim: MLP intermediate dimension. Defaults to (input_dim + output_dim) // 2.
        activation: Activation function name ('gelu', 'relu', 'silu').
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 4096,
        n_tokens: int = 32,
        hidden_dim: int | None = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self._n_output_tokens = n_tokens
        hidden = hidden_dim or (input_dim + output_dim) // 2
        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]()
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            act_fn,
            nn.Linear(hidden, output_dim),
        )

    @property
    def n_output_tokens(self) -> int:
        return self._n_output_tokens

    def forward(
        self,
        sensor_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project sensor embeddings to LLM token space.

        Args:
            sensor_embeddings: Shape (B, N_patches, input_dim).
            attention_mask: Unused by this adapter.

        Returns:
            Token embeddings of shape (B, n_output_tokens, output_dim).
        """
        # Pool N_patches → n_output_tokens
        x = sensor_embeddings.transpose(1, 2)  # (B, input_dim, N)
        x = self.pool(x)                        # (B, input_dim, n_tokens)
        x = x.transpose(1, 2)                  # (B, n_tokens, input_dim)
        return self.mlp(x)                      # (B, n_tokens, output_dim)
