"""Vision Transformer (ViT) encoder for sensor images."""

from __future__ import annotations

import torch
import torch.nn as nn

from sensorllm.models.encoders.base import SensorEncoder


class ViTSensorEncoder(SensorEncoder):
    """ViT encoder that wraps a HuggingFace vision model.

    Loads a pretrained ViT (or any compatible HF vision model) and extracts
    patch embeddings from its hidden states. The [CLS] token is discarded;
    all patch tokens are returned.

    Args:
        model_name_or_path: HuggingFace model ID (e.g., 'google/vit-base-patch16-224').
        freeze: If True, freeze all encoder parameters during training.
        output_hidden_layer: Which transformer layer to extract features from (-1 = last).
    """

    def __init__(
        self,
        model_name_or_path: str = "google/vit-base-patch16-224",
        freeze: bool = True,
        output_hidden_layer: int = -1,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self._output_hidden_layer = output_hidden_layer
        self._output_dim: int | None = None
        self.vit: nn.Module | None = None
        # Lazy initialization — call load() before forward()

    def load(self) -> None:
        """Load pretrained weights from HuggingFace hub."""
        from transformers import AutoModel

        self.vit = AutoModel.from_pretrained(self.model_name_or_path)
        self._output_dim = self.vit.config.hidden_size
        if self.freeze_params:
            for param in self.vit.parameters():
                param.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        if self._output_dim is None:
            raise RuntimeError("Call load() before accessing output_dim")
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch embeddings from sensor images.

        Args:
            images: Sensor image tensor of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, N_patches, output_dim).
        """
        raise NotImplementedError("ViTSensorEncoder.forward() not yet implemented")
