"""Optimizer and learning rate scheduler factories."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01) -> AdamW:
    """Build AdamW optimizer with weight decay applied only to non-bias parameters.

    Args:
        model: Model to optimize.
        lr: Peak learning rate.
        weight_decay: L2 regularization weight. Not applied to biases/LayerNorm.

    Returns:
        Configured AdamW optimizer.
    """
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and "bias" not in n and "norm" not in n.lower()]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and ("bias" in n or "norm" in n.lower())]
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(param_groups, lr=lr)


def build_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Build a cosine annealing LR schedule with linear warmup.

    Args:
        optimizer: Optimizer to attach schedule to.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.

    Returns:
        LambdaLR scheduler.
    """
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
