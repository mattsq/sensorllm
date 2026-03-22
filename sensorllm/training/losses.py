"""Custom loss functions for sensor-LLM training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard cross-entropy loss with token masking.

    Prompt tokens are masked (labels=-100) so loss is computed only on
    answer tokens. This is the default loss for instruction fine-tuning.

    Args:
        logits: Model output logits (B, seq_len, vocab_size).
        labels: Target token IDs (B, seq_len); -100 = masked from loss.
        ignore_index: Token ID to ignore in loss computation.

    Returns:
        Scalar loss tensor.
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )
