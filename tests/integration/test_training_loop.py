"""Integration smoke test: full forward + backward pass through the model stack."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.slow
class TestForwardBackwardSmoke:
    """Smoke tests for the full model forward pass.

    Marked as slow — skipped with: pytest -m "not slow"
    These tests require torch but do NOT require GPU or real data.
    """

    def test_linear_projection_adapter_forward_backward(self):
        """End-to-end: encoder stub → LinearProjection → fake LLM loss backward."""
        from sensorllm.models.adapters.linear_projection import LinearProjectionAdapter

        adapter = LinearProjectionAdapter(input_dim=32, output_dim=64, n_tokens=4)
        fake_encoder_out = torch.randn(2, 16, 32, requires_grad=False)
        token_embs = adapter(fake_encoder_out)
        assert token_embs.shape == (2, 4, 64)

        # Simulate a loss and backward pass
        fake_loss = token_embs.mean()
        fake_loss.backward()

        # Check adapter parameters received gradients
        for name, param in adapter.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_qformer_adapter_forward_backward(self):
        from sensorllm.models.adapters.qformer import QFormerAdapter

        adapter = QFormerAdapter(
            input_dim=32, output_dim=64, n_query_tokens=4,
            qformer_hidden_dim=32, n_heads=2, n_layers=1
        )
        x = torch.randn(2, 10, 32)
        out = adapter(x)
        assert out.shape == (2, 4, 64)
        out.mean().backward()
        for name, param in adapter.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
