#!/usr/bin/env python3
"""Single-sample inference / interactive demo for SensorLLM.

Usage:
    python scripts/infer.py \\
        --checkpoint outputs/runs/exp001_linear_proj_vit/best_model/ \\
        --sensor-file data/raw/flight_001.h5 \\
        --prompt "Describe any anomalies in the vibration sensor data."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SensorLLM inference on a single sample")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--sensor-file", required=True, help="Path to raw sensor data file")
    parser.add_argument("--prompt", required=True, help="Text prompt for the model")
    parser.add_argument("--sensor-type", default="vibration", help="Sensor modality key")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sensorllm.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Loading checkpoint: %s", args.checkpoint)
    logger.info("Sensor file: %s", args.sensor_file)
    logger.info("Prompt: %s", args.prompt)

    # TODO: Load model from checkpoint, preprocess sensor file, run generate()
    raise NotImplementedError("infer.py main() not yet implemented")


if __name__ == "__main__":
    main()
