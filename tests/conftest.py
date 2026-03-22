"""Pytest fixtures for SensorLLM test suite.

Provides lightweight synthetic data and minimal model stubs so tests run
without requiring real sensor data files or GPU resources.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ─── Synthetic sensor signals ─────────────────────────────────────────────────

@pytest.fixture
def synthetic_vibration_signal() -> np.ndarray:
    """1D synthetic vibration signal (1 second at 4096 Hz)."""
    t = np.linspace(0, 1.0, 4096, dtype=np.float32)
    # Sum of two sinusoids + noise
    return (
        np.sin(2 * np.pi * 50 * t)
        + 0.5 * np.sin(2 * np.pi * 120 * t)
        + 0.1 * np.random.randn(4096).astype(np.float32)
    )


@pytest.fixture
def synthetic_multichannel_signal() -> np.ndarray:
    """3-channel synthetic signal (n_samples=4096, n_channels=3)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4096, 3)).astype(np.float32)


@pytest.fixture
def synthetic_sensor_image() -> torch.Tensor:
    """Single synthetic sensor image tensor (1, 3, 224, 224)."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def synthetic_sensor_batch() -> torch.Tensor:
    """Batch of 4 synthetic sensor images (4, 3, 224, 224)."""
    return torch.randn(4, 3, 224, 224)


# ─── Synthetic token sequences ─────────────────────────────────────────────────

@pytest.fixture
def synthetic_token_batch() -> dict[str, torch.Tensor]:
    """Minimal tokenized batch for dataset/model tests."""
    B, seq_len = 4, 64
    return {
        "input_ids": torch.randint(0, 32000, (B, seq_len)),
        "attention_mask": torch.ones(B, seq_len, dtype=torch.long),
        "labels": torch.randint(-100, 32000, (B, seq_len)),
    }


# ─── Minimal adapter config ────────────────────────────────────────────────────

@pytest.fixture
def linear_adapter_config() -> dict:
    return {"input_dim": 768, "output_dim": 512, "n_tokens": 8}


@pytest.fixture
def qformer_adapter_config() -> dict:
    return {
        "input_dim": 64,
        "output_dim": 128,
        "n_query_tokens": 4,
        "qformer_hidden_dim": 64,
        "n_heads": 2,
        "n_layers": 1,
    }
