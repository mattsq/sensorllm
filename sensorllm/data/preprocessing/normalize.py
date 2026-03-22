"""Signal normalization utilities."""

from __future__ import annotations

import numpy as np


def normalize_signal(signal: np.ndarray, min_val: float | None = None, max_val: float | None = None) -> np.ndarray:
    """Min-max normalize a signal to [0, 1].

    Args:
        signal: Input array of any shape.
        min_val: Minimum value for normalization. If None, uses signal minimum.
        max_val: Maximum value for normalization. If None, uses signal maximum.

    Returns:
        Normalized array of same shape, dtype float32.
    """
    signal = signal.astype(np.float32)
    lo = min_val if min_val is not None else signal.min()
    hi = max_val if max_val is not None else signal.max()
    if abs(hi - lo) < 1e-8:
        return np.zeros_like(signal)
    return (signal - lo) / (hi - lo)


def zscore_normalize(signal: np.ndarray, mean: float | None = None, std: float | None = None) -> np.ndarray:
    """Z-score normalize a signal to zero mean and unit variance.

    Args:
        signal: Input array of any shape.
        mean: Mean for normalization. If None, computed from signal.
        std: Standard deviation. If None, computed from signal.

    Returns:
        Normalized array of same shape, dtype float32.
    """
    signal = signal.astype(np.float32)
    mu = mean if mean is not None else signal.mean()
    sigma = std if std is not None else signal.std()
    if sigma < 1e-8:
        return np.zeros_like(signal)
    return (signal - mu) / sigma
