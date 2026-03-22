"""1D dilated residual CNN encoder for raw sensor time-series."""

from __future__ import annotations

import torch
import torch.nn as nn

from sensorllm.models.encoders.base import SensorEncoder


class _DilatedResBlock(nn.Module):
    """1D dilated residual block with two conv layers."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class CNN1DSensorEncoder(SensorEncoder):
    """Dilated 1D residual CNN encoder for direct time-series encoding.

    Processes raw sensor signals without any image transformation. The architecture
    uses strided convolutions for downsampling and dilated residual blocks for
    multi-scale temporal feature extraction.

    Each strided conv layer reduces the sequence length by `stride`, so the total
    downsampling factor is `stride ** n_stride_layers`. The output sequence of
    temporal patches feeds directly into the sensor adapter.

    Args:
        in_channels: Number of input sensor channels (e.g., 1 for single-axis,
            3 for 3-axis IMU).
        hidden_dim: Number of channels in all intermediate feature maps and output.
        n_res_blocks: Number of dilated residual blocks per stride layer.
        n_stride_layers: Number of strided downsampling layers.
        kernel_size: Convolution kernel size for all layers.
        stride: Stride of the downsampling convolutions.
        dilation_base: Base for exponentially increasing dilation in residual blocks
            (e.g., 2 → dilations 1, 2, 4, 8).

    Shape:
        Input:  (B, in_channels, L)
        Output: (B, L // stride**n_stride_layers, hidden_dim)
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 256,
        n_res_blocks: int = 2,
        n_stride_layers: int = 4,
        kernel_size: int = 7,
        stride: int = 4,
        dilation_base: int = 2,
    ) -> None:
        super().__init__()
        self._output_dim = hidden_dim

        # Initial projection: in_channels → hidden_dim
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Stack of downsampling layers, each followed by dilated res blocks
        layers: list[nn.Module] = []
        for i in range(n_stride_layers):
            # Strided conv to downsample sequence length
            layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=stride, padding=kernel_size // 2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                )
            )
            # Dilated residual blocks for multi-scale context
            for j in range(n_res_blocks):
                dilation = dilation_base**j
                layers.append(_DilatedResBlock(hidden_dim, kernel_size=3, dilation=dilation))

        self.backbone = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(hidden_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Encode sensor signals to temporal patch embeddings.

        Args:
            signals: Shape (B, C, L) — batch of windowed sensor signals.

        Returns:
            Patch embeddings of shape (B, N_patches, hidden_dim).
        """
        x = self.input_proj(signals)    # (B, hidden_dim, L)
        x = self.backbone(x)            # (B, hidden_dim, N_patches)
        x = x.transpose(1, 2)          # (B, N_patches, hidden_dim)
        return self.out_norm(x)
