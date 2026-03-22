#!/usr/bin/env python3
"""Batch preprocess raw sensor files into spectrogram images.

Usage:
    python scripts/preprocess_sensors.py \\
        --input-dir data/raw/ \\
        --output-dir data/spectrograms/ \\
        --transform spectrogram \\
        --sensor-type vibration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw sensor files into images")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw sensor files")
    parser.add_argument("--output-dir", required=True, help="Output directory for image arrays")
    parser.add_argument("--transform", default="spectrogram", help="Transform key from TRANSFORM_REGISTRY")
    parser.add_argument("--sensor-type", default="vibration", help="Sensor modality key")
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--hop-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sensorllm.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Preprocessing %s files from: %s", args.sensor_type, args.input_dir)
    logger.info("Transform: %s", args.transform)
    logger.info("Output: %s", args.output_dir)

    # TODO: Discover sensor files, apply transform, save .npy arrays with metadata JSONL
    raise NotImplementedError("preprocess_sensors.py main() not yet implemented")


if __name__ == "__main__":
    main()
