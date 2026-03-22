#!/usr/bin/env python3
"""Evaluation entry point for SensorLLM experiments.

Usage:
    python scripts/evaluate.py \\
        --config configs/experiments/exp001_linear_proj_vit.yaml \\
        --checkpoint outputs/runs/exp001_linear_proj_vit/2024-01-15_1400/best_model/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SensorLLM checkpoint")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None, help="Path to write results JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sensorllm.utils.config import load_config
    from sensorllm.utils.logging import get_logger

    config = load_config(args.config)
    logger = get_logger(__name__)
    logger.info("Evaluating: %s on split=%s", config.get("experiment_name"), args.split)
    logger.info("Checkpoint: %s", args.checkpoint)

    # TODO: Load model, dataset, and evaluator, then call evaluator.evaluate()
    raise NotImplementedError(
        "evaluate.py main() not yet implemented — load model and evaluator here"
    )


if __name__ == "__main__":
    main()
