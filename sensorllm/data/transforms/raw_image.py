"""Raw reshape transform: directly map sensor time-series to a 2D pixel grid."""

from __future__ import annotations

import numpy as np

from sensorllm.data.transforms.base import BaseTransform


class RawImageTransform(BaseTransform):
    """Reshape a 1D signal into a 2D image by simple row-major reshaping.

    This is the simplest possible transform and serves as a no-information-loss
    baseline. The signal is zero-padded or truncated to fill the target image,
    then reshaped to (height, width).

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
    """

    def __init__(self, height: int = 64, width: int = 64) -> None:
        self.height = height
        self.width = width

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Reshape signal to a 2D image.

        Args:
            signal: 1D array of shape (n_samples,) or (n_samples, n_channels).

        Returns:
            Image of shape (height, width) or (C, height, width), float32 in [0, 1].
        """
        if signal.ndim == 2:
            channels = [self._reshape_channel(signal[:, c]) for c in range(signal.shape[1])]
            return np.stack(channels, axis=0)
        return self._reshape_channel(signal)

    def _reshape_channel(self, signal: np.ndarray) -> np.ndarray:
        target_len = self.height * self.width
        flat = signal.astype(np.float32).ravel()
        if len(flat) < target_len:
            flat = np.pad(flat, (0, target_len - len(flat)))
        else:
            flat = flat[:target_len]
        return self._normalize(flat.reshape(self.height, self.width))
