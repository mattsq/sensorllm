"""Unit tests for HFCausalLMBackbone."""

from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from sensorllm.models.llm.hf_causal_lm import HFCausalLMBackbone


@pytest.fixture(scope="module")
def tiny_gpt2_path(tmp_path_factory):
    """Create a tiny GPT-2 model locally (no download required)."""
    path = tmp_path_factory.mktemp("models") / "tiny-gpt2"
    config = GPT2Config(
        vocab_size=1000,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=128,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(str(path))
    return str(path)


@pytest.fixture(scope="module")
def backbone(tiny_gpt2_path):
    llm = HFCausalLMBackbone(
        model_name_or_path=tiny_gpt2_path,
        freeze=True,
        torch_dtype="float32",
        device_map="cpu",
    )
    llm.load()
    return llm


class TestHFCausalLMBackbone:
    def test_hidden_size(self, backbone):
        assert backbone.hidden_size == 64

    def test_get_input_embeddings(self, backbone):
        emb = backbone.get_input_embeddings()
        assert emb.embedding_dim == 64

    def test_forward_shape(self, backbone):
        x = torch.randn(1, 10, 64)
        logits, loss = backbone(x)
        assert logits.shape == (1, 10, 1000)
        assert loss is None

    def test_forward_with_labels(self, backbone):
        x = torch.randn(1, 10, 64)
        labels = torch.randint(0, 1000, (1, 10))
        logits, loss = backbone(x, labels=labels)
        assert logits.shape == (1, 10, 1000)
        assert loss is not None
        assert loss.dim() == 0

    def test_frozen_params(self, backbone):
        for param in backbone.model.parameters():
            assert not param.requires_grad

    def test_generate(self, backbone):
        x = torch.randn(1, 5, 64)
        out = backbone.generate(x, max_new_tokens=3)
        assert out.shape[0] == 1
        assert out.shape[1] >= 3


class TestHFCausalLMBackboneErrors:
    def test_forward_before_load(self):
        llm = HFCausalLMBackbone("nonexistent-model")
        with pytest.raises(RuntimeError, match="Call load"):
            llm(torch.randn(1, 10, 64))

    def test_hidden_size_before_load(self):
        llm = HFCausalLMBackbone("nonexistent-model")
        with pytest.raises(RuntimeError, match="Call load"):
            _ = llm.hidden_size

    def test_get_input_embeddings_before_load(self):
        llm = HFCausalLMBackbone("nonexistent-model")
        with pytest.raises(RuntimeError, match="Call load"):
            llm.get_input_embeddings()
