"""Unit tests for adapter architectures."""

from __future__ import annotations

import pytest
import torch

from sensorllm.models.adapters import ADAPTER_REGISTRY
from sensorllm.models.adapters.base import SensorAdapter
from sensorllm.models.adapters.linear_projection import LinearProjectionAdapter
from sensorllm.models.adapters.qformer import QFormerAdapter
from sensorllm.models.adapters.perceiver import PerceiverResamplerAdapter


class TestLinearProjectionAdapter:
    def test_output_shape(self):
        adapter = LinearProjectionAdapter(input_dim=64, output_dim=128, n_tokens=8)
        x = torch.randn(2, 16, 64)  # (B, N_patches, encoder_dim) — from time-series encoder
        out = adapter(x)
        assert out.shape == (2, 8, 128)

    def test_n_output_tokens_property(self):
        adapter = LinearProjectionAdapter(input_dim=64, output_dim=128, n_tokens=16)
        assert adapter.n_output_tokens == 16

    def test_output_dtype(self):
        adapter = LinearProjectionAdapter(input_dim=32, output_dim=64, n_tokens=4)
        x = torch.randn(1, 10, 32)
        out = adapter(x)
        assert out.dtype == torch.float32

    def test_fixed_output_regardless_of_input_length(self):
        adapter = LinearProjectionAdapter(input_dim=64, output_dim=128, n_tokens=8)
        for n_patches in [4, 16, 64, 126]:  # various patch counts from different encoders
            x = torch.randn(1, n_patches, 64)
            out = adapter(x)
            assert out.shape == (1, 8, 128), f"Failed for n_patches={n_patches}"


class TestQFormerAdapter:
    def test_output_shape(self):
        adapter = QFormerAdapter(
            input_dim=64, output_dim=128, n_query_tokens=4,
            qformer_hidden_dim=64, n_heads=2, n_layers=1
        )
        x = torch.randn(2, 10, 64)
        out = adapter(x)
        assert out.shape == (2, 4, 128)

    def test_n_output_tokens_property(self):
        adapter = QFormerAdapter(input_dim=64, output_dim=128, n_query_tokens=6,
                                  qformer_hidden_dim=64, n_heads=2, n_layers=1)
        assert adapter.n_output_tokens == 6

    def test_with_attention_mask(self):
        adapter = QFormerAdapter(input_dim=64, output_dim=128, n_query_tokens=4,
                                  qformer_hidden_dim=64, n_heads=2, n_layers=1)
        x = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10, dtype=torch.long)
        mask[0, 8:] = 0
        out = adapter(x, attention_mask=mask)
        assert out.shape == (2, 4, 128)


class TestPerceiverResamplerAdapter:
    def test_output_shape(self):
        adapter = PerceiverResamplerAdapter(
            input_dim=64, output_dim=128, n_latents=4,
            latent_dim=32, n_heads=2, n_layers=1
        )
        x = torch.randn(2, 20, 64)  # longer N from PatchTST encoder
        out = adapter(x)
        assert out.shape == (2, 4, 128)

    def test_n_output_tokens_property(self):
        adapter = PerceiverResamplerAdapter(input_dim=64, output_dim=128, n_latents=8,
                                             latent_dim=32, n_heads=2, n_layers=1)
        assert adapter.n_output_tokens == 8

    def test_handles_long_input_sequence(self):
        """Perceiver should handle the ~126 patches from PatchTST on a 4096-sample window."""
        adapter = PerceiverResamplerAdapter(
            input_dim=64, output_dim=128, n_latents=32,
            latent_dim=32, n_heads=2, n_layers=1
        )
        x = torch.randn(2, 126, 64)  # PatchTST output length
        out = adapter(x)
        assert out.shape == (2, 32, 128)


class TestAdapterRegistry:
    def test_all_keys_registered(self):
        expected = {"linear_projection", "qformer", "perceiver", "mlp_mixer"}
        assert expected == set(ADAPTER_REGISTRY.keys())

    def test_all_classes_inherit_base(self):
        for name, cls in ADAPTER_REGISTRY.items():
            assert issubclass(cls, SensorAdapter), f"{name} does not inherit SensorAdapter"
