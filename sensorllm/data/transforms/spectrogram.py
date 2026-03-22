"""Mel spectrogram transform for sensor time-series."""

from __future__ import annotations

import numpy as np

from sensorllm.data.transforms.base import BaseTransform


class MelSpectrogramTransform(BaseTransform):
    """Convert a 1D sensor signal to a mel spectrogram image.

    Uses librosa to compute a short-time Fourier transform (STFT) mapped onto
    a mel filterbank. Output is log-power mel spectrogram normalized to [0, 1].

    Args:
        sample_rate: Signal sampling frequency in Hz.
        n_mels: Number of mel filter banks (image height).
        n_fft: FFT window size in samples.
        hop_length: STFT hop size in samples. Determines image width.
        fmin: Minimum frequency for mel filterbank in Hz.
        fmax: Maximum frequency for mel filterbank in Hz (None = sample_rate/2).
    """

    def __init__(
        self,
        sample_rate: float = 400.0,
        n_mels: int = 64,
        n_fft: int = 512,
        hop_length: int = 64,
        fmin: float = 0.0,
        fmax: float | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from a 1D signal.

        Args:
            signal: 1D array of shape (n_samples,) or (n_samples, n_channels).

        Returns:
            Mel spectrogram of shape (n_mels, T) or (C, n_mels, T), float32 in [0, 1].
        """
        try:
            import librosa
        except ImportError as e:
            raise ImportError("librosa is required for MelSpectrogramTransform") from e

        if signal.ndim == 2:
            channels = [self._compute_mel(signal[:, c], librosa) for c in range(signal.shape[1])]
            return np.stack(channels, axis=0)
        return self._compute_mel(signal, librosa)

    def _compute_mel(self, signal: np.ndarray, librosa) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=signal.astype(np.float32),
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return self._normalize(log_mel)
