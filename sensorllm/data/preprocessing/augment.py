"""Signal augmentation utilities for training data diversity."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(signal: np.ndarray, snr_db: float = 20.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add Gaussian white noise to a signal at the specified SNR.

    Args:
        signal: Input signal array.
        snr_db: Signal-to-noise ratio in decibels.
        rng: Random number generator. If None, uses default numpy RNG.

    Returns:
        Noisy signal array of same shape and dtype.
    """
    rng = rng or np.random.default_rng()
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), signal.shape)
    return (signal + noise).astype(signal.dtype)


def time_warp(signal: np.ndarray, max_warp: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply random time warping to a 1D signal.

    Stretches or compresses the signal in the time dimension by a random factor,
    then resamples back to the original length.

    Args:
        signal: 1D input signal of shape (n_samples,).
        max_warp: Maximum fractional warp magnitude (e.g., 0.1 = ±10%).
        rng: Random number generator.

    Returns:
        Time-warped signal of same length.
    """
    rng = rng or np.random.default_rng()
    warp = 1.0 + rng.uniform(-max_warp, max_warp)
    n = len(signal)
    warped_len = max(1, int(n * warp))
    original_indices = np.linspace(0, n - 1, warped_len)
    warped = np.interp(original_indices, np.arange(n), signal.astype(np.float32))
    resampled_indices = np.linspace(0, warped_len - 1, n)
    return np.interp(resampled_indices, np.arange(warped_len), warped).astype(signal.dtype)
