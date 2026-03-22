"""ResNet encoder for sensor images."""

from __future__ import annotations

import torch
import torch.nn as nn

from sensorllm.models.encoders.base import SensorEncoder


class ResNetSensorEncoder(SensorEncoder):
    """ResNet encoder that uses feature map spatial tokens.

    Extracts features from an intermediate ResNet layer and flattens the
    spatial dimensions to produce a sequence of patch embeddings.

    Args:
        model_name: torchvision ResNet variant ('resnet50', 'resnet34', etc.).
        pretrained: If True, load ImageNet pretrained weights.
        freeze: If True, freeze all parameters during training.
        feature_layer: Which layer to extract features from ('layer3' or 'layer4').
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        freeze: bool = True,
        feature_layer: str = "layer4",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_params = freeze
        self.feature_layer = feature_layer
        self._output_dim: int | None = None
        self.backbone: nn.Module | None = None

    @property
    def output_dim(self) -> int:
        if self._output_dim is None:
            raise RuntimeError("Call load() before accessing output_dim")
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature embeddings from sensor images.

        Args:
            images: Sensor image tensor of shape (B, C, H, W).

        Returns:
            Feature embeddings of shape (B, H'*W', output_dim).
        """
        raise NotImplementedError("ResNetSensorEncoder.forward() not yet implemented")
