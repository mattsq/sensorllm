"""Reproducibility helpers: seed setting and determinism."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy, PyTorch (CPU and CUDA),
    and enables cuDNN determinism.

    Args:
        seed: Integer seed value. Use the `seed` field from experiment config.

    Example:
        from sensorllm.utils.reproducibility import set_seed
        set_seed(config["seed"])
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
