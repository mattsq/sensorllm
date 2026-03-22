"""Sliding window and event-based segmentation utilities."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np


def sliding_windows(
    signal: np.ndarray,
    window_size: int,
    hop_size: int,
    drop_last: bool = True,
) -> Generator[np.ndarray, None, None]:
    """Generate overlapping windows from a signal.

    Args:
        signal: 1D or 2D array of shape (n_samples,) or (n_samples, n_channels).
        window_size: Number of samples per window.
        hop_size: Number of samples to advance between windows.
        drop_last: If True, drop the final incomplete window.

    Yields:
        Windows of shape (window_size,) or (window_size, n_channels).
    """
    n = signal.shape[0]
    start = 0
    while start + window_size <= n:
        yield signal[start : start + window_size]
        start += hop_size
    if not drop_last and start < n:
        yield signal[start:]


def segment_by_event(
    signal: np.ndarray,
    event_indices: list[int],
    context_before: int = 0,
    context_after: int = 0,
) -> list[np.ndarray]:
    """Extract signal segments centered around event timestamps.

    Args:
        signal: 1D or 2D signal array.
        event_indices: Sample indices where events of interest occur.
        context_before: Number of samples to include before each event.
        context_after: Number of samples to include after each event.

    Returns:
        List of signal segments, one per event index.
    """
    n = signal.shape[0]
    segments = []
    for idx in event_indices:
        start = max(0, idx - context_before)
        end = min(n, idx + context_after + 1)
        segments.append(signal[start:end])
    return segments
