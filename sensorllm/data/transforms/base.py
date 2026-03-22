"""Abstract base class for signal-to-image transforms."""

from __future__ import annotations

import abc

import numpy as np


class BaseTransform(abc.ABC):
    """Abstract base class for sensor signal → 2D image transforms.

    All transforms are stateless callables; all parameters are set at construction
    time. Multi-channel inputs are handled by applying the transform per channel
    and stacking along a leading channel dimension.

    Contract:
        Input:  np.ndarray of shape (n_samples,) or (n_samples, n_channels)
        Output: np.ndarray of shape (H, W) or (C, H, W), dtype float32, values in [0, 1]

    Example:
        transform = MelSpectrogramTransform(sample_rate=400, n_mels=64)
        signal = np.random.randn(4096)
        image = transform(signal)   # shape (64, W), values in [0, 1]
    """

    @abc.abstractmethod
    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Transform a 1D sensor signal into a 2D image array.

        Args:
            signal: Input signal of shape (n_samples,) or (n_samples, n_channels).

        Returns:
            Image array of shape (H, W) for single channel, or (C, H, W) for
            multi-channel input. dtype=float32, values in [0, 1].
        """
        raise NotImplementedError

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Min-max normalize an array to [0, 1].

        Args:
            arr: Input array of any shape.

        Returns:
            Normalized array with same shape, dtype float32.
        """
        arr = arr.astype(np.float32)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
