"""HuggingFace AutoModelForCausalLM backbone wrapper."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from sensorllm.models.llm.base import LLMBackbone

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


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
        model_name_or_path: str = "",
        freeze: bool = True,
        lora_config: dict | None = None,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.freeze = freeze
        self.lora_config = lora_config
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self._hidden_size: int | None = None
        self.model: PreTrainedModel | None = None

    @classmethod
    def from_model(cls, model: PreTrainedModel, freeze: bool = True) -> HFCausalLMBackbone:
        """Create a backbone from an already-instantiated HuggingFace model.

        Useful for testing with tiny randomly-initialized models.
        """
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.model_name_or_path = ""
        instance.freeze = freeze
        instance.lora_config = None
        instance.torch_dtype = "float32"
        instance.device_map = "cpu"
        instance.model = model
        instance._hidden_size = model.config.hidden_size
        if freeze:
            model.requires_grad_(False)
        return instance

    def load(self) -> None:
        """Load model weights from HuggingFace hub or local path."""
        dtype = _DTYPE_MAP.get(self.torch_dtype, torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype,
            device_map=self.device_map,
        )
        self._hidden_size = self.model.config.hidden_size
        logger.info(
            "Loaded LLM %s (hidden_size=%d)", self.model_name_or_path, self._hidden_size
        )
        if self.freeze:
            self.model.requires_grad_(False)
            logger.info("Froze all LLM parameters")

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
        if self.model is None:
            raise RuntimeError("Call load() before forward()")
        output = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output.logits, output.loss

    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Call load() before generate()")
        return self.model.generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
