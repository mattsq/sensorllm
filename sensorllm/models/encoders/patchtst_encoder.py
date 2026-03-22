"""PatchTST-style encoder for sensor time-series (Nie et al. 2023)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sensorllm.models.encoders.base import SensorEncoder


class PatchTSTSensorEncoder(SensorEncoder):
    """PatchTST-inspired encoder for direct time-series encoding.

    Key design choices from PatchTST (Nie et al. 2023):
    - **Channel independence**: each sensor channel is patched and encoded
      separately, then representations are aggregated. This improves
      generalization across sensor types with different channel counts.
    - **Overlapping patches**: patches are extracted with configurable stride,
      allowing overlap between adjacent patches for smoother temporal coverage.
    - **Transformer backbone**: standard encoder with pre-LN for stability.

    The output is the mean of per-channel patch representations, giving a
    unified (B, N_patches, d_model) tensor regardless of input channel count.

    Args:
        in_channels: Number of sensor input channels.
        patch_len: Patch length in samples.
        stride: Stride between consecutive patches (< patch_len = overlap).
        d_model: Transformer hidden dimension and output size.
        n_heads: Number of self-attention heads.
        n_layers: Number of Transformer encoder layers.
        dim_feedforward: Feed-forward network size. Defaults to 4 * d_model.
        dropout: Dropout probability.

    Shape:
        Input:  (B, in_channels, L)
        Output: (B, N_patches, d_model)
                where N_patches = floor((L - patch_len) / stride) + 1
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_len: int = 64,
        stride: int = 32,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.in_channels = in_channels
        self._output_dim = d_model

        # Per-channel: patch_len → d_model
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Learned positional embedding (max 1024 patches per channel)
        self.pos_embed = nn.Embedding(1024, d_model)

        ff_dim = dim_feedforward or d_model * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(d_model)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract overlapping patches from a single-channel signal.

        Args:
            x: Single-channel signal (B, L).

        Returns:
            Patches (B, N_patches, patch_len).
        """
        B, L = x.shape
        # Use unfold for efficient overlapping patch extraction
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches: (B, N_patches, patch_len)
        return patches

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Encode sensor signals to temporal patch embeddings.

        Args:
            signals: Shape (B, C, L).

        Returns:
            Patch embeddings of shape (B, N_patches, d_model).
        """
        B, C, L = signals.shape

        channel_outputs = []
        for c in range(C):
            ch = signals[:, c, :]           # (B, L)
            patches = self._extract_patches(ch)  # (B, N, patch_len)
            N = patches.size(1)

            x = self.patch_embed(patches)   # (B, N, d_model)
            positions = torch.arange(N, device=x.device).unsqueeze(0)
            x = x + self.pos_embed(positions)  # (B, N, d_model)
            x = self.transformer(x)         # (B, N, d_model)
            channel_outputs.append(x)

        # Channel aggregation: mean over channels
        out = torch.stack(channel_outputs, dim=0).mean(dim=0)  # (B, N, d_model)
        return self.out_norm(out)
