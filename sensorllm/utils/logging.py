"""Structured logging setup with optional W&B integration."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a named logger with consistent formatting.

    Always use this instead of bare print() statements. The logger name should
    be the module's __name__ for traceable log attribution.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default: INFO).

    Returns:
        Configured Logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Starting training run: %s", experiment_name)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def init_wandb(config: dict[str, Any], run_name: str | None = None) -> None:
    """Initialize a W&B run with the given experiment config.

    Reads WANDB_PROJECT and WANDB_ENTITY from environment variables.

    Args:
        config: Experiment config dict (will be logged as W&B config).
        run_name: Optional run display name. Defaults to config['experiment_name'].
    """
    try:
        import wandb
    except ImportError:
        get_logger(__name__).warning("wandb not installed; skipping W&B initialization")
        return

    project = os.environ.get("WANDB_PROJECT", "sensorllm")
    entity = os.environ.get("WANDB_ENTITY")
    name = run_name or config.get("experiment_name")
    wandb.init(project=project, entity=entity, name=name, config=config)
