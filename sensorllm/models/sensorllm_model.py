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
        1. SensorEncoder: raw sensor signal -> temporal patch embeddings
        2. SensorAdapter: patch embeddings -> LLM token embeddings (fixed length)
        3. LLMBackbone: token embeddings -> text (causal LM)

    Sensor data is encoded directly from raw time-series -- no image conversion.

    Sensor token embeddings are prepended to the text token embeddings before
    the LLM forward pass.

    Args:
        encoder: Instantiated SensorEncoder (CNN1D, Transformer, PatchTST, etc.).
        adapter: Instantiated SensorAdapter (Linear, Q-Former, Perceiver, etc.).
        llm: Instantiated LLMBackbone.
        sensor_token_id: Token ID of the <sensor> placeholder in the tokenizer vocab.

    Example:
        model = SensorLLMModel(encoder, adapter, llm, sensor_token_id=32000)
        loss = model(sensor_signals, input_ids, attention_mask, labels)
        generated = model.generate(sensor_signals, prompt_ids, prompt_mask)
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
        sensor_signals: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Full forward pass: sensor signal + text tokens -> loss/logits.

        Sensor token embeddings are prepended to the text embeddings along the
        sequence dimension. Labels at sensor positions are set to -100 (ignored).

        Args:
            sensor_signals: Batch of windowed sensor signals (B, C, L).
            input_ids: Tokenized text prompt + answer (B, seq_len).
            attention_mask: Text attention mask (B, seq_len).
            labels: Target token IDs for loss (B, seq_len); -100 = masked.

        Returns:
            Tuple of (logits, loss). Loss is None when labels is None.
        """
        B = sensor_signals.shape[0]
        device = sensor_signals.device

        # Encode sensor signals to LLM-space token embeddings
        sensor_embs = self._encode_sensor(sensor_signals)  # (B, T, D_llm)
        T = sensor_embs.shape[1]

        # Get text embeddings from the LLM's embedding layer
        text_embs = self.llm.get_input_embeddings()(input_ids)  # (B, seq_len, D_llm)

        # Prepend sensor embeddings to text embeddings
        combined_embs = torch.cat([sensor_embs, text_embs], dim=1)

        # Build combined attention mask
        sensor_mask = torch.ones(B, T, device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([sensor_mask, attention_mask], dim=1)

        # Build combined labels (ignore sensor token positions)
        combined_labels = None
        if labels is not None:
            sensor_labels = torch.full((B, T), -100, device=device, dtype=labels.dtype)
            combined_labels = torch.cat([sensor_labels, labels], dim=1)

        logits, loss = self.llm(
            inputs_embeds=combined_embs,
            attention_mask=combined_mask,
            labels=combined_labels,
        )
        return logits, loss

    def generate(
        self,
        sensor_signals: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Generate text conditioned on sensor signals and a text prompt.

        Args:
            sensor_signals: (B, C, L) sensor signal batch.
            prompt_ids: Tokenized prompt (B, prompt_len).
            prompt_mask: Prompt attention mask (B, prompt_len).
            **generation_kwargs: Forwarded to LLM generate() (max_new_tokens, etc.).

        Returns:
            Generated token IDs (B, output_len).
        """
        B = sensor_signals.shape[0]
        device = sensor_signals.device

        sensor_embs = self._encode_sensor(sensor_signals)  # (B, T, D_llm)
        T = sensor_embs.shape[1]

        prompt_embs = self.llm.get_input_embeddings()(prompt_ids)
        combined_embs = torch.cat([sensor_embs, prompt_embs], dim=1)

        sensor_mask = torch.ones(B, T, device=device, dtype=prompt_mask.dtype)
        combined_mask = torch.cat([sensor_mask, prompt_mask], dim=1)

        return self.llm.generate(
            inputs_embeds=combined_embs,
            attention_mask=combined_mask,
            **generation_kwargs,
        )

    def _encode_sensor(self, sensor_signals: torch.Tensor) -> torch.Tensor:
        """Encode raw sensor signals to LLM-space token embeddings.

        Args:
            sensor_signals: (B, C, L) -- batch of windowed sensor signals.

        Returns:
            Token embeddings (B, n_output_tokens, llm_hidden_size).
        """
        patch_embs = self.encoder(sensor_signals)   # (B, N, encoder_dim)
        token_embs = self.adapter(patch_embs)        # (B, n_tokens, llm_dim)
        return token_embs
