"""LLM backbone wrappers."""

from sensorllm.models.llm.base import LLMBackbone
from sensorllm.models.llm.hf_causal_lm import HFCausalLMBackbone

__all__ = ["LLMBackbone", "HFCausalLMBackbone"]
