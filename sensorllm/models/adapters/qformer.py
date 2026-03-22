"""Q-Former adapter — BLIP-2 style cross-attention with learnable queries."""

from __future__ import annotations

import torch
import torch.nn as nn

from sensorllm.models.adapters.base import SensorAdapter


class QFormerAdapter(SensorAdapter):
    """Q-Former (Querying Transformer) adapter, inspired by BLIP-2.

    Maintains a set of learnable query tokens that attend to sensor encoder
    embeddings via cross-attention. The query outputs are then projected to
    the LLM hidden dimension. This approach allows selective extraction of
    task-relevant features from the sensor representation.

    Args:
        input_dim: Dimensionality of sensor encoder embeddings.
        output_dim: LLM hidden dimension.
        n_query_tokens: Number of learnable query tokens (= n_output_tokens).
        qformer_hidden_dim: Hidden size within the Q-Former transformer.
        n_heads: Number of attention heads.
        n_layers: Number of Q-Former transformer layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 4096,
        n_query_tokens: int = 32,
        qformer_hidden_dim: int = 768,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._n_output_tokens = n_query_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, n_query_tokens, qformer_hidden_dim))
        self.input_proj = nn.Linear(input_dim, qformer_hidden_dim)
        encoder_layer = nn.TransformerDecoderLayer(
            d_model=qformer_hidden_dim,
            nhead=n_heads,
            dim_feedforward=qformer_hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(qformer_hidden_dim, output_dim)

    @property
    def n_output_tokens(self) -> int:
        return self._n_output_tokens

    def forward(
        self,
        sensor_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-attend learnable queries to sensor embeddings.

        Args:
            sensor_embeddings: Shape (B, N_patches, input_dim).
            attention_mask: Optional key padding mask (B, N_patches).

        Returns:
            Token embeddings of shape (B, n_query_tokens, output_dim).
        """
        B = sensor_embeddings.size(0)
        memory = self.input_proj(sensor_embeddings)  # (B, N, qformer_dim)
        queries = self.query_tokens.expand(B, -1, -1)  # (B, n_tokens, qformer_dim)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # True = ignore
        out = self.transformer(queries, memory, memory_key_padding_mask=key_padding_mask)
        return self.output_proj(out)  # (B, n_tokens, output_dim)
