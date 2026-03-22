"""Integration smoke test: full forward + backward pass through the model stack."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.slow
class TestForwardBackwardSmoke:
    """Smoke tests for end-to-end encoder → adapter forward + backward pass.

    Marked as slow — skip with: pytest -m "not slow"
    These tests require torch but do NOT require GPU or real data.
    """

    def test_cnn1d_linear_forward_backward(self):
        """CNN1D encoder → LinearProjection adapter: end-to-end gradient flow."""
        from sensorllm.models.encoders.cnn1d_encoder import CNN1DSensorEncoder
        from sensorllm.models.adapters.linear_projection import LinearProjectionAdapter

        enc = CNN1DSensorEncoder(in_channels=1, hidden_dim=32, n_stride_layers=2, stride=4)
        adapter = LinearProjectionAdapter(input_dim=32, output_dim=64, n_tokens=4)

        signals = torch.randn(2, 1, 256)         # (B, C, L)
        latents = enc(signals)                    # (B, N, 32)
        token_embs = adapter(latents)             # (B, 4, 64)
        assert token_embs.shape == (2, 4, 64)

        token_embs.mean().backward()
        for name, param in list(enc.named_parameters()) + list(adapter.named_parameters()):
            assert param.grad is not None, f"No gradient for {name}"

    def test_transformer_qformer_forward_backward(self):
        """Transformer encoder → Q-Former adapter: end-to-end gradient flow."""
        from sensorllm.models.encoders.transformer_encoder import TransformerSensorEncoder
        from sensorllm.models.adapters.qformer import QFormerAdapter

        enc = TransformerSensorEncoder(
            in_channels=1, patch_size=16, d_model=32, n_heads=2, n_layers=1
        )
        adapter = QFormerAdapter(
            input_dim=32, output_dim=64, n_query_tokens=4,
            qformer_hidden_dim=32, n_heads=2, n_layers=1
        )

        signals = torch.randn(2, 1, 64)          # N = 64/16 = 4 patches
        latents = enc(signals)                   # (B, 4, 32)
        token_embs = adapter(latents)            # (B, 4, 64)
        assert token_embs.shape == (2, 4, 64)

        token_embs.mean().backward()
        for name, param in list(enc.named_parameters()) + list(adapter.named_parameters()):
            assert param.grad is not None, f"No gradient for {name}"

    def test_patchtst_perceiver_forward_backward(self):
        """PatchTST encoder → Perceiver adapter: end-to-end gradient flow."""
        from sensorllm.models.encoders.patchtst_encoder import PatchTSTSensorEncoder
        from sensorllm.models.adapters.perceiver import PerceiverResamplerAdapter

        enc = PatchTSTSensorEncoder(
            in_channels=1, patch_len=16, stride=8, d_model=32, n_heads=2, n_layers=1
        )
        adapter = PerceiverResamplerAdapter(
            input_dim=32, output_dim=64, n_latents=4,
            latent_dim=32, n_heads=2, n_layers=1
        )

        signals = torch.randn(2, 1, 64)          # N = floor((64-16)/8)+1 = 7
        latents = enc(signals)                   # (B, 7, 32)
        token_embs = adapter(latents)            # (B, 4, 64)
        assert token_embs.shape == (2, 4, 64)

        token_embs.mean().backward()
        for name, param in list(enc.named_parameters()) + list(adapter.named_parameters()):
            assert param.grad is not None, f"No gradient for {name}"
