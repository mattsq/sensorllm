#!/usr/bin/env python3
"""Training entry point for SensorLLM experiments.

Usage:
    python scripts/train.py --config configs/experiments/exp001_linear_proj_vit.yaml
    python scripts/train.py --config configs/experiments/exp001_linear_proj_vit.yaml \\
        --override training.learning_rate=5e-5 training.max_steps=20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SensorLLM model")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dot-notation config overrides (e.g. training.learning_rate=1e-4)",
    )
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sensorllm.utils.config import load_config
    from sensorllm.utils.logging import get_logger, init_wandb
    from sensorllm.utils.reproducibility import set_seed

    overrides = {}
    for item in args.override:
        key, _, val = item.partition("=")
        overrides[key] = val

    config = load_config(args.config, overrides=overrides if overrides else None)
    logger = get_logger(__name__)
    logger.info("Starting experiment: %s", config.get("experiment_name"))
    logger.info("Config loaded from: %s", args.config)

    set_seed(config.get("seed", 42))
    init_wandb(config)

    # TODO: Build model, datasets, and trainer, then call trainer.train()
    raise NotImplementedError(
        "train.py main() not yet implemented — build model, datasets, and trainer here"
    )


if __name__ == "__main__":
    main()
