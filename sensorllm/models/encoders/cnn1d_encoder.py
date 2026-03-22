"""1D CNN encoder for raw time-series (bypasses image transform)."""

from __future__ import annotations

import torch
import torch.nn as nn

from sensorllm.models.encoders.base import SensorEncoder


class CNN1DSensorEncoder(SensorEncoder):
    """Baseline 1D CNN encoder that operates directly on raw time-series.

    Unlike ViT and ResNet encoders, this encoder skips the image transform
    and processes raw 1D signals, providing a non-image-adapter baseline.

    Args:
        in_channels: Number of sensor input channels.
        hidden_dim: Number of channels in CNN feature maps.
        n_layers: Number of 1D convolutional layers.
        kernel_size: Convolution kernel size.
        stride: Convolution stride (determines output sequence length).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 4,
        kernel_size: int = 7,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self._output_dim = hidden_dim
        layers = []
        ch_in = in_channels
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(ch_in, hidden_dim, kernel_size, stride=stride, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ]
            ch_in = hidden_dim
        self.cnn = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of 1D sensor signals.

        Args:
            images: Signal tensor of shape (B, C, L) — treat H*W as sequence length.

        Returns:
            Embedding tensor of shape (B, L', output_dim).
        """
        x = self.cnn(images)  # (B, hidden_dim, L')
        return x.transpose(1, 2)  # (B, L', hidden_dim)
