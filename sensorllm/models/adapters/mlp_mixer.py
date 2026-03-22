"""MLP-Mixer adapter — token mixing without attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from sensorllm.models.adapters.base import SensorAdapter


class MLPMixerAdapter(SensorAdapter):
    """MLP-Mixer-based adapter for sensor-to-LLM token compression.

    Applies alternating token-mixing and channel-mixing MLPs (MLP-Mixer style)
    to sensor embeddings, then pools to a fixed number of output tokens.
    Avoids attention entirely — computationally efficient for long sensor sequences.

    Args:
        input_dim: Dimensionality of sensor encoder embeddings.
        output_dim: LLM hidden dimension.
        n_output_tokens: Number of output tokens to produce.
        n_layers: Number of MLP-Mixer layers.
        token_mixing_dim: Hidden dim for token-mixing MLP.
        channel_mixing_dim: Hidden dim for channel-mixing MLP.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 4096,
        n_output_tokens: int = 32,
        n_layers: int = 4,
        token_mixing_dim: int = 256,
        channel_mixing_dim: int = 1024,
    ) -> None:
        super().__init__()
        self._n_output_tokens = n_output_tokens
        self.layers = nn.ModuleList([
            _MixerLayer(input_dim, token_mixing_dim, channel_mixing_dim) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(input_dim)
        self.pool = nn.AdaptiveAvgPool1d(n_output_tokens)
        self.output_proj = nn.Linear(input_dim, output_dim)

    @property
    def n_output_tokens(self) -> int:
        return self._n_output_tokens

    def forward(
        self,
        sensor_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mix and project sensor embeddings.

        Args:
            sensor_embeddings: Shape (B, N_patches, input_dim).
            attention_mask: Unused.

        Returns:
            Token embeddings of shape (B, n_output_tokens, output_dim).
        """
        x = sensor_embeddings  # (B, N, D)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.transpose(1, 2)               # (B, D, N)
        x = self.pool(x)                    # (B, D, n_tokens)
        x = x.transpose(1, 2)              # (B, n_tokens, D)
        return self.output_proj(x)          # (B, n_tokens, output_dim)


class _MixerLayer(nn.Module):
    """Single MLP-Mixer layer: token mixing + channel mixing with residuals."""

    def __init__(self, dim: int, token_dim: int, channel_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # Token mixing: operates across the token (sequence) dimension
        self.token_mix = nn.Sequential(nn.LazyLinear(token_dim), nn.GELU(), nn.LazyLinear(0))
        # Channel mixing: operates across the channel (feature) dimension
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, channel_dim), nn.GELU(), nn.Linear(channel_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing
        residual = x
        x = self.norm1(x).transpose(1, 2)  # (B, D, N)
        N = x.size(-1)
        token_mlp = nn.Sequential(nn.Linear(N, N * 2), nn.GELU(), nn.Linear(N * 2, N)).to(x.device)
        x = token_mlp(x).transpose(1, 2)   # (B, N, D)
        x = residual + x
        # Channel mixing
        x = x + self.channel_mix(self.norm2(x))
        return x
