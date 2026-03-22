"""Continuous Wavelet Transform (scalogram) for sensor time-series."""

from __future__ import annotations

import numpy as np

from sensorllm.data.transforms.base import BaseTransform


class CWTTransform(BaseTransform):
    """Convert a 1D sensor signal to a wavelet scalogram via CWT.

    Uses scipy.signal.cwt with a Morlet wavelet. The scalogram captures
    time-frequency features with better time resolution at high frequencies
    and better frequency resolution at low frequencies compared to STFT.

    Args:
        n_scales: Number of wavelet scales (image height).
        min_scale: Minimum scale parameter.
        max_scale: Maximum scale parameter.
        wavelet: Wavelet name supported by scipy (default: 'morlet2').
    """

    def __init__(
        self,
        n_scales: int = 64,
        min_scale: float = 1.0,
        max_scale: float = 128.0,
        wavelet: str = "morlet2",
    ) -> None:
        self.n_scales = n_scales
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.wavelet = wavelet

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Compute wavelet scalogram from a 1D signal.

        Args:
            signal: 1D array of shape (n_samples,) or (n_samples, n_channels).

        Returns:
            Scalogram of shape (n_scales, n_samples) or (C, n_scales, n_samples),
            float32 in [0, 1].
        """
        if signal.ndim == 2:
            channels = [self._compute_cwt(signal[:, c]) for c in range(signal.shape[1])]
            return np.stack(channels, axis=0)
        return self._compute_cwt(signal)

    def _compute_cwt(self, signal: np.ndarray) -> np.ndarray:
        from scipy import signal as scipy_signal

        scales = np.logspace(
            np.log10(self.min_scale), np.log10(self.max_scale), num=self.n_scales
        )
        wavelet_fn = getattr(scipy_signal, self.wavelet, scipy_signal.morlet2)
        coeffs = scipy_signal.cwt(signal.astype(np.float32), wavelet_fn, scales)
        return self._normalize(np.abs(coeffs))
