"""Top-level SensorLLM model: wires encoder + adapter + LLM backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from sensorllm.models.encoders.base import SensorEncoder
from sensorllm.models.adapters.base import SensorAdapter
from sensorllm.models.llm.base import LLMBackbone


class SensorLLMModel(nn.Module):
    """End-to-end SensorLLM model.

    Composes three components:
        1. SensorEncoder: sensor image → patch embeddings
        2. SensorAdapter: patch embeddings → LLM token embeddings (fixed length)
        3. LLMBackbone: token embeddings → text (causal LM)

    Sensor token embeddings are prepended to the text token embeddings before
    the LLM forward pass. A special <sensor> placeholder token in the prompt
    marks the injection point.

    Args:
        encoder: Instantiated SensorEncoder.
        adapter: Instantiated SensorAdapter.
        llm: Instantiated LLMBackbone.
        sensor_token_id: Token ID of the <sensor> placeholder in the tokenizer vocabulary.

    Example:
        model = SensorLLMModel(encoder, adapter, llm, sensor_token_id=32000)
        loss = model(sensor_images, input_ids, attention_mask, labels)
        generated = model.generate(sensor_images, prompt_ids, prompt_mask)
    """

    def __init__(
        self,
        encoder: SensorEncoder,
        adapter: SensorAdapter,
        llm: LLMBackbone,
        sensor_token_id: int = 32000,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.llm = llm
        self.sensor_token_id = sensor_token_id

    def forward(
        self,
        sensor_images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Full forward pass: sensor image + text tokens → loss/logits.

        Args:
            sensor_images: Batch of sensor images (B, C, H, W).
            input_ids: Tokenized text prompt + answer (B, seq_len).
            attention_mask: Text attention mask (B, seq_len).
            labels: Target token IDs for loss (B, seq_len); -100 = masked.

        Returns:
            Tuple of (logits, loss). Loss is None when labels is None.
        """
        raise NotImplementedError("SensorLLMModel.forward() not yet implemented")

    def generate(
        self,
        sensor_images: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Generate text conditioned on sensor images and a text prompt.

        Args:
            sensor_images: (B, C, H, W) sensor image batch.
            prompt_ids: Tokenized prompt (B, prompt_len).
            prompt_mask: Prompt attention mask (B, prompt_len).
            **generation_kwargs: Forwarded to LLM generate() (max_new_tokens, etc.).

        Returns:
            Generated token IDs (B, output_len).
        """
        raise NotImplementedError("SensorLLMModel.generate() not yet implemented")

    def _encode_sensor(self, sensor_images: torch.Tensor) -> torch.Tensor:
        """Encode sensor images to LLM-space token embeddings.

        Args:
            sensor_images: (B, C, H, W)

        Returns:
            Token embeddings (B, n_output_tokens, llm_hidden_size)
        """
        patch_embs = self.encoder(sensor_images)   # (B, N, encoder_dim)
        token_embs = self.adapter(patch_embs)       # (B, n_tokens, llm_dim)
        return token_embs
