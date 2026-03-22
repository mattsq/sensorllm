"""Generate synthetic sensor data for end-to-end pipeline smoke testing.

Usage examples::

    # Generate with default settings (all sensors, 20 samples/class)
    python scripts/generate_synthetic_data.py

    # Specify data root and sample count
    python scripts/generate_synthetic_data.py --data-root /tmp/sensorllm_data --samples 50

    # Only vibration and IMU, 10 samples per class
    python scripts/generate_synthetic_data.py \\
        --sensors vibration imu \\
        --samples 10

    # Reproduce a previous run
    python scripts/generate_synthetic_data.py --seed 123

Output files::

    <data_root>/raw/synthetic/vibration_normal_0000.h5
    <data_root>/raw/synthetic/vibration_bearing_fault_0001.h5
    ...
    <data_root>/splits/synthetic_train.jsonl
    <data_root>/splits/synthetic_val.jsonl
    <data_root>/splits/synthetic_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow running as a script without installing the package
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sensorllm.data.synthetic.dataset_builder import SyntheticDatasetBuilder
from sensorllm.data.synthetic.sensor_generator import SensorType, SyntheticSensorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic sensor data for smoke testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("SENSORLLM_DATA_ROOT", "data"),
        help="Path to the data root directory (default: $SENSORLLM_DATA_ROOT or './data').",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples per (sensor_type, event_type) class (default: 20).",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        choices=[s.value for s in SensorType],
        default=None,
        help="Sensor modalities to generate (default: all four).",
    )
    parser.add_argument(
        "--split-ratios",
        nargs=3,
        type=float,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split fractions (must sum to 1, default: 0.7 0.15 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    # Per-sensor sample-rate / duration overrides
    parser.add_argument(
        "--vibration-sample-rate",
        type=float,
        default=None,
        help="Override sample rate (Hz) for vibration signals.",
    )
    parser.add_argument(
        "--vibration-duration",
        type=float,
        default=None,
        help="Override duration (s) for vibration signals.",
    )
    parser.add_argument(
        "--temperature-duration",
        type=float,
        default=None,
        help="Override duration (s) for temperature signals (default: 60 s).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary table of generated samples after completion.",
    )
    return parser.parse_args()


def _build_config_overrides(args: argparse.Namespace) -> dict[SensorType, SyntheticSensorConfig]:
    overrides: dict[SensorType, SyntheticSensorConfig] = {}

    from sensorllm.data.synthetic.sensor_generator import _VIBRATION_CFG, _TEMPERATURE_CFG

    if args.vibration_sample_rate is not None or args.vibration_duration is not None:
        overrides[SensorType.VIBRATION] = SyntheticSensorConfig(
            sample_rate=args.vibration_sample_rate or _VIBRATION_CFG.sample_rate,
            duration_s=args.vibration_duration or _VIBRATION_CFG.duration_s,
            n_channels=_VIBRATION_CFG.n_channels,
            noise_std=_VIBRATION_CFG.noise_std,
        )

    if args.temperature_duration is not None:
        overrides[SensorType.TEMPERATURE] = SyntheticSensorConfig(
            sample_rate=_TEMPERATURE_CFG.sample_rate,
            duration_s=args.temperature_duration,
            n_channels=_TEMPERATURE_CFG.n_channels,
            noise_std=_TEMPERATURE_CFG.noise_std,
        )

    return overrides


def print_summary(records: list[dict]) -> None:
    """Print a tabular summary of generated records."""
    from collections import Counter

    split_counts: Counter = Counter()
    sensor_counts: Counter = Counter()
    event_counts: Counter = Counter()

    for r in records:
        split_counts[r["split"]] += 1
        sensor_counts[r["sensor"]] += 1
        event_counts[f"{r['sensor']}/{r['event_type']}"] += 1

    total = len(records)
    print(f"\n{'─' * 60}")
    print(f"  Synthetic Dataset Summary  ({total} total samples)")
    print(f"{'─' * 60}")
    print(f"  {'Split':<10} {'Count':>6}  {'%':>6}")
    for split in ("train", "val", "test"):
        n = split_counts[split]
        print(f"  {split:<10} {n:>6}  {100 * n / total:>5.1f}%")
    print()
    print(f"  {'Sensor':<14} {'Count':>6}")
    for sensor, n in sorted(sensor_counts.items()):
        print(f"  {sensor:<14} {n:>6}")
    print()
    print(f"  {'Sensor/Event':<32} {'Count':>6}")
    for key, n in sorted(event_counts.items()):
        print(f"  {key:<32} {n:>6}")
    print(f"{'─' * 60}\n")


def main() -> int:
    args = parse_args()

    # Validate split ratios
    if abs(sum(args.split_ratios) - 1.0) > 1e-6:
        logger.error("--split-ratios must sum to 1.0, got %s", args.split_ratios)
        return 1

    data_root = Path(args.data_root).expanduser().resolve()
    sensor_types = (
        [SensorType(s) for s in args.sensors] if args.sensors else None
    )

    logger.info("Data root         : %s", data_root)
    logger.info("Samples per class : %d", args.samples)
    logger.info(
        "Sensors           : %s",
        [s.value for s in (sensor_types or list(SensorType))],
    )
    logger.info("Split ratios      : train=%.2f val=%.2f test=%.2f", *args.split_ratios)
    logger.info("Seed              : %d", args.seed)

    config_overrides = _build_config_overrides(args)

    builder = SyntheticDatasetBuilder(
        data_root=data_root,
        samples_per_class=args.samples,
        split_ratios=tuple(args.split_ratios),  # type: ignore[arg-type]
        sensor_types=sensor_types,
        config_overrides=config_overrides,
        seed=args.seed,
    )

    records = builder.build()

    if args.summary:
        print_summary(records)

    logger.info("Done. %d samples written to %s", len(records), data_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
