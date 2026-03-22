"""Recurrence plot transform for sensor time-series."""

from __future__ import annotations

import numpy as np

from sensorllm.data.transforms.base import BaseTransform


class RecurrencePlotTransform(BaseTransform):
    """Convert a 1D sensor signal to a recurrence plot.

    A recurrence plot encodes the distance between all pairs of trajectory points
    in a reconstructed phase space, capturing nonlinear dynamical structure that
    spectral methods miss. Useful for detecting periodic faults and chaotic behaviour.

    Args:
        dimension: Embedding dimension for phase space reconstruction.
        time_delay: Time delay for embedding (in samples).
        threshold: Distance threshold for binary recurrence plot (None = soft/grayscale).
        image_size: Downsample output to this square size (None = full size).
    """

    def __init__(
        self,
        dimension: int = 3,
        time_delay: int = 1,
        threshold: float | None = None,
        image_size: int | None = 128,
    ) -> None:
        self.dimension = dimension
        self.time_delay = time_delay
        self.threshold = threshold
        self.image_size = image_size

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Compute recurrence plot from a 1D signal.

        Args:
            signal: 1D array of shape (n_samples,) or (n_samples, n_channels).
                For multi-channel input, the first channel is used.

        Returns:
            Recurrence plot of shape (H, W), float32 in [0, 1].
        """
        if signal.ndim == 2:
            signal = signal[:, 0]

        return self._compute_rp(signal.astype(np.float32))

    def _compute_rp(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal) - (self.dimension - 1) * self.time_delay
        trajectory = np.array(
            [signal[i : i + self.dimension * self.time_delay : self.time_delay] for i in range(n)]
        )
        diff = trajectory[:, np.newaxis, :] - trajectory[np.newaxis, :, :]
        distances = np.sqrt((diff**2).sum(axis=-1))

        if self.threshold is not None:
            rp = (distances < self.threshold).astype(np.float32)
        else:
            rp = self._normalize(distances)
            rp = 1.0 - rp  # invert: close = bright

        if self.image_size is not None and rp.shape[0] != self.image_size:
            from PIL import Image

            rp_img = Image.fromarray((rp * 255).astype(np.uint8))
            rp_img = rp_img.resize((self.image_size, self.image_size), Image.BILINEAR)
            rp = np.array(rp_img).astype(np.float32) / 255.0

        return rp
