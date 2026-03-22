"""Perceiver Resampler adapter — Flamingo-style latent array cross-attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from sensorllm.models.adapters.base import SensorAdapter


class PerceiverResamplerAdapter(SensorAdapter):
    """Perceiver Resampler adapter, inspired by Flamingo.

    Uses a fixed-size latent array that cross-attends to sensor embeddings,
    compressing an arbitrary-length sensor sequence to a fixed number of tokens.
    Unlike Q-Former, the latent array also attends to itself (self-attention)
    in each layer, enabling richer latent representations.

    Args:
        input_dim: Dimensionality of sensor encoder embeddings.
        output_dim: LLM hidden dimension.
        n_latents: Number of latent tokens (= n_output_tokens).
        latent_dim: Dimensionality of the latent array.
        n_heads: Number of attention heads.
        n_layers: Number of Perceiver layers (each: self-attn + cross-attn + FFN).
        ff_mult: Feed-forward expansion multiplier.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 4096,
        n_latents: int = 64,
        latent_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self._n_output_tokens = n_latents
        self.latents = nn.Parameter(torch.randn(1, n_latents, latent_dim))
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.layers = nn.ModuleList([
            _PerceiverLayer(latent_dim, n_heads, ff_mult) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, output_dim)

    @property
    def n_output_tokens(self) -> int:
        return self._n_output_tokens

    def forward(
        self,
        sensor_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress sensor embeddings into latent tokens.

        Args:
            sensor_embeddings: Shape (B, N_patches, input_dim).
            attention_mask: Optional key padding mask (B, N_patches).

        Returns:
            Token embeddings of shape (B, n_latents, output_dim).
        """
        B = sensor_embeddings.size(0)
        context = self.input_proj(sensor_embeddings)   # (B, N, latent_dim)
        x = self.latents.expand(B, -1, -1)             # (B, n_latents, latent_dim)
        for layer in self.layers:
            x = layer(x, context, attention_mask)
        x = self.norm(x)
        return self.output_proj(x)                     # (B, n_latents, output_dim)


class _PerceiverLayer(nn.Module):
    """Single Perceiver layer: latent self-attention + cross-attention to context + FFN."""

    def __init__(self, dim: int, n_heads: int, ff_mult: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        kpm = (key_padding_mask == 0) if key_padding_mask is not None else None
        x = x + self.cross_attn(self.norm2(x), context, context, key_padding_mask=kpm)[0]
        x = x + self.ff(self.norm3(x))
        return x
