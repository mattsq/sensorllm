"""1D Transformer encoder for direct sensor time-series encoding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sensorllm.models.encoders.base import SensorEncoder


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for 1D sequences."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerSensorEncoder(SensorEncoder):
    """Transformer encoder that operates directly on sensor time-series patches.

    The signal is divided into non-overlapping temporal patches of `patch_size`
    samples. Each patch is projected to `d_model` dimensions, then processed by
    a stack of standard Transformer encoder layers with positional encodings.

    This architecture captures long-range temporal dependencies across patches
    while being computationally tractable (sequence length = L / patch_size).

    Args:
        in_channels: Number of input sensor channels.
        patch_size: Number of time samples per patch. Must divide the window length
            evenly. Determines N_patches = L / patch_size.
        d_model: Transformer hidden dimension and output embedding size.
        n_heads: Number of self-attention heads. Must divide d_model.
        n_layers: Number of Transformer encoder layers.
        dim_feedforward: Feed-forward network hidden size. Defaults to 4 * d_model.
        dropout: Dropout probability in attention and feed-forward layers.
        positional_encoding: 'sinusoidal' (fixed) or 'learned'.

    Shape:
        Input:  (B, in_channels, L)
        Output: (B, L // patch_size, d_model)
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 64,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        positional_encoding: str = "sinusoidal",
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self._output_dim = d_model

        # Project each flattened patch (in_channels * patch_size) → d_model
        self.patch_embed = nn.Linear(in_channels * patch_size, d_model)

        if positional_encoding == "sinusoidal":
            self.pos_enc = _SinusoidalPositionalEncoding(d_model, dropout=dropout)
        else:
            # Learned: max 1024 patches
            self.pos_embed = nn.Embedding(1024, d_model)
            self.pos_enc = None

        self._use_learned_pos = positional_encoding == "learned"
        ff_dim = dim_feedforward or d_model * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(d_model)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Encode sensor signals to temporal patch embeddings.

        Args:
            signals: Shape (B, C, L). L must be divisible by patch_size.

        Returns:
            Patch embeddings of shape (B, N_patches, d_model).
        """
        B, C, L = signals.shape
        if L % self.patch_size != 0:
            raise ValueError(
                f"Signal length {L} is not divisible by patch_size {self.patch_size}. "
                f"Pad or trim the signal to a multiple of {self.patch_size}."
            )
        N = L // self.patch_size

        # Reshape into patches: (B, N, C * patch_size)
        x = signals.reshape(B, C, N, self.patch_size)   # (B, C, N, P)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C * self.patch_size)  # (B, N, C*P)

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, d_model)

        # Positional encoding
        if self._use_learned_pos:
            positions = torch.arange(N, device=x.device).unsqueeze(0)
            x = x + self.pos_embed(positions)
        else:
            x = self.pos_enc(x)

        # Transformer encoder
        x = self.transformer(x)   # (B, N, d_model)
        return self.out_norm(x)
