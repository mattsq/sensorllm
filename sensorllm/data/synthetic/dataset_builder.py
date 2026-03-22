"""Assembles the synthetic dataset: writes HDF5 sensor files and JSONL index files.

Output layout (relative to ``data_root``)::

    data_root/
        raw/
            synthetic/
                vibration_normal_000.h5
                vibration_bearing_fault_001.h5
                ...
        splits/
            synthetic_train.jsonl
            synthetic_val.jsonl
            synthetic_test.jsonl

Each HDF5 file has the structure::

    /sensor_type      str scalar
    /event_type       str scalar
    /signal           float32 (n_samples, n_channels)
    /sample_rate      float64 scalar
    /metadata/        group
        flight_id     str scalar
        label         str scalar

Each JSONL line has the schema expected by the dataset classes::

    {
        "path": "raw/synthetic/vibration_normal_000.h5",
        "sensor": "vibration",
        "event_type": "normal",
        "split": "train",
        "label": "normal",
        "description": "<pretrain description>",
        "qa_pairs": [{"question": "...", "answer": "..."}]
    }
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import h5py
import numpy as np

from sensorllm.data.synthetic.annotation_generator import AnnotationGenerator
from sensorllm.data.synthetic.sensor_generator import (
    DEFAULT_CONFIGS,
    VALID_EVENTS,
    EventType,
    SensorType,
    SyntheticSensorConfig,
    generate_signal,
)

logger = logging.getLogger(__name__)


class SyntheticDatasetBuilder:
    """Generates synthetic sensor data files and annotation index files.

    Args:
        data_root: Root data directory (corresponds to ``SENSORLLM_DATA_ROOT``).
        samples_per_class: Number of samples to generate per
            (sensor_type, event_type) combination.
        split_ratios: Fraction of data for train/val/test splits (must sum to 1).
        sensor_types: Which sensor modalities to include.
            Defaults to all four: vibration, imu, temperature, pressure.
        config_overrides: Optional per-sensor-type config overrides.
            Keys are SensorType values; values are SyntheticSensorConfig instances.
        seed: Global random seed for reproducibility.

    Example::

        builder = SyntheticDatasetBuilder(
            data_root=Path("data"),
            samples_per_class=50,
        )
        index = builder.build()
        print(f"Generated {len(index)} samples")
    """

    def __init__(
        self,
        data_root: Path,
        samples_per_class: int = 20,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        sensor_types: list[SensorType] | None = None,
        config_overrides: dict[SensorType, SyntheticSensorConfig] | None = None,
        seed: int = 42,
    ) -> None:
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {split_ratios}")

        self.data_root = Path(data_root)
        self.samples_per_class = samples_per_class
        self.split_ratios = split_ratios
        self.sensor_types = sensor_types or list(SensorType)
        self.config_overrides = config_overrides or {}
        self.seed = seed
        self._annotation_gen = AnnotationGenerator()

    # ─── Public API ──────────────────────────────────────────────────────────

    def build(self) -> list[dict]:
        """Generate all sensor files and index files.

        Returns:
            Full index as a list of record dicts (same as written to JSONL files).
        """
        rng = random.Random(self.seed)
        raw_dir = self.data_root / "raw" / "synthetic"
        splits_dir = self.data_root / "splits"
        raw_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)

        all_records: list[dict] = []

        for sensor_type in self.sensor_types:
            for event_type in VALID_EVENTS[sensor_type]:
                config = self.config_overrides.get(sensor_type, DEFAULT_CONFIGS[sensor_type])
                records = self._generate_class(
                    sensor_type=sensor_type,
                    event_type=event_type,
                    config=config,
                    raw_dir=raw_dir,
                    rng=rng,
                )
                all_records.extend(records)
                logger.info(
                    "Generated %d samples for %s/%s",
                    len(records),
                    sensor_type.value,
                    event_type.value,
                )

        # Shuffle all records before splitting
        rng.shuffle(all_records)

        # Assign splits
        n = len(all_records)
        n_train = int(self.split_ratios[0] * n)
        n_val = int(self.split_ratios[1] * n)
        for i, record in enumerate(all_records):
            if i < n_train:
                record["split"] = "train"
            elif i < n_train + n_val:
                record["split"] = "val"
            else:
                record["split"] = "test"

        # Write JSONL split files
        self._write_splits(all_records, splits_dir)

        logger.info(
            "Synthetic dataset complete: %d total samples in %s",
            n,
            self.data_root,
        )
        return all_records

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _generate_class(
        self,
        sensor_type: SensorType,
        event_type: EventType,
        config: SyntheticSensorConfig,
        raw_dir: Path,
        rng: random.Random,
    ) -> list[dict]:
        """Generate ``samples_per_class`` HDF5 files for one (sensor, event) class."""
        records = []
        for i in range(self.samples_per_class):
            # Each sample gets its own seed for reproducibility but uniqueness
            sample_seed = rng.randint(0, 2**31 - 1)
            per_sample_config = SyntheticSensorConfig(
                sample_rate=config.sample_rate,
                duration_s=config.duration_s,
                n_channels=config.n_channels,
                noise_std=config.noise_std,
                rng_seed=sample_seed,
            )
            signal = generate_signal(sensor_type, event_type, per_sample_config)

            # Build filename: e.g., vibration_bearing_fault_007.h5
            stem = f"{sensor_type.value}_{event_type.value}_{i:04d}"
            h5_path = raw_dir / f"{stem}.h5"
            flight_id = f"synthetic_{stem}"

            self._write_h5(
                path=h5_path,
                signal=signal,
                sample_rate=config.sample_rate,
                sensor_type=sensor_type,
                event_type=event_type,
                flight_id=flight_id,
            )

            # Generate annotations
            description = self._annotation_gen.pretrain_description(
                sensor_type, event_type, signal, config.sample_rate
            )
            qa_pairs = self._annotation_gen.qa_pairs(
                sensor_type, event_type, signal, config.sample_rate
            )

            # Relative path (as stored in JSONL index)
            rel_path = str(h5_path.relative_to(self.data_root))

            records.append(
                {
                    "path": rel_path,
                    "sensor": sensor_type.value,
                    "event_type": event_type.value,
                    "split": "train",  # placeholder; overwritten after shuffle
                    "label": event_type.value,
                    "sample_rate": config.sample_rate,
                    "n_channels": config.n_channels,
                    "n_samples": signal.shape[0],
                    "flight_id": flight_id,
                    "description": description,
                    "qa_pairs": qa_pairs,
                }
            )
        return records

    @staticmethod
    def _write_h5(
        path: Path,
        signal: np.ndarray,
        sample_rate: float,
        sensor_type: SensorType,
        event_type: EventType,
        flight_id: str,
    ) -> None:
        """Write a single synthetic sample to an HDF5 file."""
        with h5py.File(path, "w") as f:
            f.create_dataset("signal", data=signal, compression="gzip", compression_opts=4)
            f.create_dataset("sample_rate", data=sample_rate)
            f.attrs["sensor_type"] = sensor_type.value
            f.attrs["event_type"] = event_type.value
            grp = f.create_group("metadata")
            grp.attrs["flight_id"] = flight_id
            grp.attrs["label"] = event_type.value

    @staticmethod
    def _write_splits(records: list[dict], splits_dir: Path) -> None:
        """Write train/val/test JSONL index files."""
        split_files: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for rec in records:
            split_files[rec["split"]].append(rec)

        for split, recs in split_files.items():
            out_path = splits_dir / f"synthetic_{split}.jsonl"
            with out_path.open("w") as fh:
                for rec in recs:
                    fh.write(json.dumps(rec) + "\n")
            logger.info("Wrote %d records → %s", len(recs), out_path)
