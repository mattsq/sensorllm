#!/usr/bin/env python3
"""Export a trained SensorLLM model for deployment.

Usage:
    python scripts/export_model.py \\
        --checkpoint outputs/runs/exp001_linear_proj_vit/best_model/ \\
        --output-dir exports/exp001_v1/ \\
        --format huggingface
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a SensorLLM model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", required=True, help="Export output directory")
    parser.add_argument("--format", default="huggingface", choices=["huggingface", "onnx"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sensorllm.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Exporting checkpoint: %s", args.checkpoint)
    logger.info("Format: %s → %s", args.format, args.output_dir)

    # TODO: Load model, merge LoRA weights, save in target format
    raise NotImplementedError("export_model.py main() not yet implemented")


if __name__ == "__main__":
    main()
