"""HuggingFace AutoModelForCausalLM backbone wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn

from sensorllm.models.llm.base import LLMBackbone


class HFCausalLMBackbone(LLMBackbone):
    """Wrapper around any HuggingFace AutoModelForCausalLM.

    Supports optional LoRA fine-tuning via PEFT. The LLM's token embedding
    layer is exposed for sensor token injection.

    Args:
        model_name_or_path: HuggingFace model ID or local checkpoint path.
        freeze: If True, freeze all base model parameters (use with LoRA).
        lora_config: PEFT LoraConfig dict; if None, LoRA is not applied.
        torch_dtype: Model dtype ('float16', 'bfloat16', 'float32').
        device_map: HuggingFace device_map for model loading ('auto', 'cpu', etc.).
    """

    def __init__(
        self,
        model_name_or_path: str,
        freeze: bool = True,
        lora_config: dict | None = None,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self._hidden_size: int | None = None
        self.model: nn.Module | None = None
        # Lazy initialization — call load() before use

    def load(self) -> None:
        """Load model weights from HuggingFace hub."""
        raise NotImplementedError("HFCausalLMBackbone.load() not yet implemented")

    def get_input_embeddings(self) -> nn.Embedding:
        if self.model is None:
            raise RuntimeError("Call load() before get_input_embeddings()")
        return self.model.get_input_embeddings()

    @property
    def hidden_size(self) -> int:
        if self._hidden_size is None:
            raise RuntimeError("Call load() before accessing hidden_size")
        return self._hidden_size

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("HFCausalLMBackbone.forward() not yet implemented")

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("HFCausalLMBackbone.generate() not yet implemented")
