"""Unit tests for the synthetic data pipeline.

Tests cover:
- Signal shape and dtype for all (sensor_type, event_type) combinations
- Annotation generation (pretrain descriptions + QA pairs)
- Dataset builder output (HDF5 files + JSONL index files)
- CLI script dry-run via subprocess
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from sensorllm.data.synthetic.annotation_generator import AnnotationGenerator, compute_stats
from sensorllm.data.synthetic.dataset_builder import SyntheticDatasetBuilder
from sensorllm.data.synthetic.sensor_generator import (
    DEFAULT_CONFIGS,
    VALID_EVENTS,
    EventType,
    SensorType,
    SyntheticSensorConfig,
    generate_imu_signal,
    generate_pressure_signal,
    generate_temperature_signal,
    generate_vibration_signal,
    generate_signal,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

_SMALL_VIBRATION_CFG = SyntheticSensorConfig(
    sample_rate=512.0, duration_s=0.5, n_channels=1, noise_std=0.01, rng_seed=0
)
_SMALL_IMU_CFG = SyntheticSensorConfig(
    sample_rate=100.0, duration_s=0.5, n_channels=6, noise_std=0.01, rng_seed=0
)
_SMALL_TEMP_CFG = SyntheticSensorConfig(
    sample_rate=1.0, duration_s=10.0, n_channels=1, noise_std=0.1, rng_seed=0
)
_SMALL_PRESSURE_CFG = SyntheticSensorConfig(
    sample_rate=50.0, duration_s=0.5, n_channels=1, noise_std=0.01, rng_seed=0
)


# ─── Sensor generator tests ───────────────────────────────────────────────────


class TestSyntheticSensorConfig:
    def test_n_samples_computed(self):
        cfg = SyntheticSensorConfig(sample_rate=100.0, duration_s=2.0)
        assert cfg.n_samples == 200

    def test_default_values(self):
        cfg = SyntheticSensorConfig()
        assert cfg.rng_seed is None
        assert cfg.n_channels == 1


class TestVibrationGenerator:
    @pytest.mark.parametrize(
        "event_type",
        [EventType.NORMAL, EventType.BEARING_FAULT, EventType.IMBALANCE, EventType.MISALIGNMENT],
    )
    def test_output_shape(self, event_type):
        sig = generate_vibration_signal(event_type=event_type, config=_SMALL_VIBRATION_CFG)
        expected_n = _SMALL_VIBRATION_CFG.n_samples
        assert sig.shape == (expected_n, 1), f"Expected ({expected_n}, 1), got {sig.shape}"

    def test_output_dtype(self):
        sig = generate_vibration_signal(config=_SMALL_VIBRATION_CFG)
        assert sig.dtype == np.float32

    def test_multichannel(self):
        cfg = SyntheticSensorConfig(sample_rate=512.0, duration_s=0.5, n_channels=3, rng_seed=0)
        sig = generate_vibration_signal(config=cfg)
        assert sig.shape == (cfg.n_samples, 3)

    def test_reproducible_with_seed(self):
        cfg1 = SyntheticSensorConfig(sample_rate=512.0, duration_s=0.5, rng_seed=99)
        cfg2 = SyntheticSensorConfig(sample_rate=512.0, duration_s=0.5, rng_seed=99)
        sig1 = generate_vibration_signal(config=cfg1)
        sig2 = generate_vibration_signal(config=cfg2)
        np.testing.assert_array_equal(sig1, sig2)

    def test_bearing_fault_has_higher_peak(self):
        """Bearing fault impulses should produce a higher peak than normal."""
        cfg = SyntheticSensorConfig(sample_rate=512.0, duration_s=0.5, rng_seed=7)
        normal_sig = generate_vibration_signal(EventType.NORMAL, config=cfg)
        fault_sig = generate_vibration_signal(EventType.BEARING_FAULT, config=cfg)
        assert np.max(np.abs(fault_sig)) > np.max(np.abs(normal_sig))

    def test_imbalance_higher_rms(self):
        """Imbalance adds amplitude at 1× so RMS should be higher than normal."""
        cfg = SyntheticSensorConfig(sample_rate=512.0, duration_s=0.5, rng_seed=3)
        normal_sig = generate_vibration_signal(EventType.NORMAL, config=cfg)
        imbal_sig = generate_vibration_signal(EventType.IMBALANCE, config=cfg)
        assert np.sqrt(np.mean(imbal_sig**2)) > np.sqrt(np.mean(normal_sig**2))


class TestIMUGenerator:
    @pytest.mark.parametrize(
        "event_type",
        [EventType.NORMAL, EventType.TURBULENCE, EventType.UNUSUAL_ATTITUDE],
    )
    def test_output_shape(self, event_type):
        sig = generate_imu_signal(event_type=event_type, config=_SMALL_IMU_CFG)
        assert sig.shape == (_SMALL_IMU_CFG.n_samples, 6)

    def test_output_dtype(self):
        sig = generate_imu_signal(config=_SMALL_IMU_CFG)
        assert sig.dtype == np.float32

    def test_turbulence_has_higher_std(self):
        """Turbulence should increase signal variance."""
        cfg = SyntheticSensorConfig(sample_rate=100.0, duration_s=0.5, n_channels=6, rng_seed=5)
        normal_sig = generate_imu_signal(EventType.NORMAL, config=cfg)
        turb_sig = generate_imu_signal(EventType.TURBULENCE, config=cfg)
        assert turb_sig.std() > normal_sig.std()

    def test_unusual_attitude_gravity_shift(self):
        """Unusual attitude should shift the z-accelerometer value."""
        cfg = SyntheticSensorConfig(sample_rate=100.0, duration_s=0.5, n_channels=6, rng_seed=5)
        normal_sig = generate_imu_signal(EventType.NORMAL, config=cfg)
        attitude_sig = generate_imu_signal(EventType.UNUSUAL_ATTITUDE, config=cfg)
        # z-accelerometer (channel 5) should be reduced from 9.81
        assert attitude_sig[:, 5].mean() < normal_sig[:, 5].mean()


class TestTemperatureGenerator:
    @pytest.mark.parametrize(
        "event_type",
        [EventType.NORMAL, EventType.OVERHEAT, EventType.RAPID_COOLING],
    )
    def test_output_shape(self, event_type):
        sig = generate_temperature_signal(event_type=event_type, config=_SMALL_TEMP_CFG)
        assert sig.shape == (_SMALL_TEMP_CFG.n_samples, 1)

    def test_output_dtype(self):
        sig = generate_temperature_signal(config=_SMALL_TEMP_CFG)
        assert sig.dtype == np.float32

    def test_overheat_monotonically_increasing(self):
        cfg = SyntheticSensorConfig(sample_rate=1.0, duration_s=10.0, n_channels=1, rng_seed=0)
        sig = generate_temperature_signal(EventType.OVERHEAT, config=cfg)
        # The mean of the second half should exceed the mean of the first half
        half = cfg.n_samples // 2
        assert sig[half:, 0].mean() > sig[:half, 0].mean()

    def test_rapid_cooling_decreases(self):
        cfg = SyntheticSensorConfig(sample_rate=1.0, duration_s=10.0, n_channels=1, rng_seed=0)
        sig = generate_temperature_signal(EventType.RAPID_COOLING, config=cfg)
        # The latter portion should be cooler
        assert sig[-1, 0] < sig[0, 0]


class TestPressureGenerator:
    @pytest.mark.parametrize(
        "event_type",
        [EventType.NORMAL, EventType.PRESSURE_LOSS, EventType.SPIKE],
    )
    def test_output_shape(self, event_type):
        sig = generate_pressure_signal(event_type=event_type, config=_SMALL_PRESSURE_CFG)
        assert sig.shape == (_SMALL_PRESSURE_CFG.n_samples, 1)

    def test_output_dtype(self):
        sig = generate_pressure_signal(config=_SMALL_PRESSURE_CFG)
        assert sig.dtype == np.float32

    def test_pressure_loss_decreasing(self):
        sig = generate_pressure_signal(EventType.PRESSURE_LOSS, config=_SMALL_PRESSURE_CFG)
        half = _SMALL_PRESSURE_CFG.n_samples // 2
        assert sig[half:, 0].mean() < sig[:half, 0].mean()

    def test_spike_peak_above_normal(self):
        # Use a larger config so the Hanning spike envelope has enough samples to be non-zero.
        cfg = SyntheticSensorConfig(sample_rate=100.0, duration_s=1.0, n_channels=1, rng_seed=0)
        sig_normal = generate_pressure_signal(EventType.NORMAL, config=cfg)
        sig_spike = generate_pressure_signal(EventType.SPIKE, config=cfg)
        assert np.max(sig_spike) > np.max(sig_normal)


class TestGenerateSignalDispatch:
    def test_valid_dispatch(self):
        sig = generate_signal(SensorType.VIBRATION, EventType.NORMAL, _SMALL_VIBRATION_CFG)
        assert sig.dtype == np.float32

    def test_invalid_event_raises(self):
        with pytest.raises(ValueError, match="not valid for sensor"):
            generate_signal(SensorType.VIBRATION, EventType.TURBULENCE)

    @pytest.mark.parametrize("sensor_type", list(SensorType))
    def test_all_sensors_dispatch(self, sensor_type):
        event_type = VALID_EVENTS[sensor_type][0]
        sig = generate_signal(sensor_type, event_type)
        assert isinstance(sig, np.ndarray)
        assert sig.dtype == np.float32


# ─── Annotation tests ─────────────────────────────────────────────────────────


class TestComputeStats:
    def test_rms_positive(self):
        sig = np.ones((100, 1), dtype=np.float32)
        stats = compute_stats(sig, sample_rate=100.0)
        assert abs(stats.rms - 1.0) < 1e-5

    def test_dominant_freq_detected(self):
        sr = 100.0
        t = np.linspace(0, 1.0, int(sr), endpoint=False)
        sig = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)[:, None]
        stats = compute_stats(sig, sample_rate=sr)
        assert stats.dominant_freq_hz is not None
        # Should detect roughly 10 Hz (allow ±2 Hz tolerance)
        assert abs(stats.dominant_freq_hz - 10.0) < 2.0


class TestAnnotationGenerator:
    def setup_method(self):
        self.gen = AnnotationGenerator()

    @pytest.mark.parametrize("sensor_type", list(SensorType))
    def test_pretrain_description_nonempty(self, sensor_type):
        event_type = VALID_EVENTS[sensor_type][0]
        cfg = DEFAULT_CONFIGS[sensor_type]
        sig = generate_signal(sensor_type, event_type, cfg)
        desc = self.gen.pretrain_description(sensor_type, event_type, sig, cfg.sample_rate)
        assert isinstance(desc, str)
        assert len(desc) > 20

    @pytest.mark.parametrize("sensor_type", list(SensorType))
    def test_qa_pairs_structure(self, sensor_type):
        event_type = VALID_EVENTS[sensor_type][0]
        cfg = DEFAULT_CONFIGS[sensor_type]
        sig = generate_signal(sensor_type, event_type, cfg)
        pairs = self.gen.qa_pairs(sensor_type, event_type, sig, cfg.sample_rate)
        assert len(pairs) >= 3
        for pair in pairs:
            assert "question" in pair
            assert "answer" in pair
            assert len(pair["question"]) > 5
            assert len(pair["answer"]) > 5

    def test_format_pretrain_prompt(self):
        instruction, response = self.gen.format_pretrain_prompt("Normal vibration signal.")
        assert isinstance(instruction, str)
        assert isinstance(response, str)
        assert "Normal vibration signal." in response

    def test_format_qa_prompt(self):
        instruction, response = self.gen.format_qa_prompt(
            question="Is there a fault?", answer="No fault detected."
        )
        assert "Is there a fault?" in instruction
        assert response == "No fault detected."

    def test_all_sensor_event_combos_have_annotations(self):
        """Every (sensor_type, event_type) must produce non-empty descriptions and QA."""
        for sensor_type in SensorType:
            for event_type in VALID_EVENTS[sensor_type]:
                cfg = DEFAULT_CONFIGS[sensor_type]
                sig = generate_signal(sensor_type, event_type, cfg)
                desc = self.gen.pretrain_description(sensor_type, event_type, sig, cfg.sample_rate)
                pairs = self.gen.qa_pairs(sensor_type, event_type, sig, cfg.sample_rate)
                assert len(desc) > 0, f"Empty description for {sensor_type}/{event_type}"
                assert len(pairs) > 0, f"No QA pairs for {sensor_type}/{event_type}"


# ─── Dataset builder tests ────────────────────────────────────────────────────


class TestSyntheticDatasetBuilder:
    """Integration tests: verify that files are written and indices are correct."""

    @pytest.fixture
    def tiny_builder(self, tmp_path):
        """Builder configured for fast tests: 2 samples, vibration only."""
        small_config = SyntheticSensorConfig(
            sample_rate=512.0, duration_s=0.1, n_channels=1, rng_seed=0
        )
        return SyntheticDatasetBuilder(
            data_root=tmp_path,
            samples_per_class=2,
            sensor_types=[SensorType.VIBRATION],
            config_overrides={SensorType.VIBRATION: small_config},
            seed=0,
        )

    def test_build_returns_records(self, tiny_builder):
        records = tiny_builder.build()
        # 4 vibration event types × 2 samples each = 8
        assert len(records) == 8

    def test_h5_files_created(self, tiny_builder, tmp_path):
        tiny_builder.build()
        h5_files = list((tmp_path / "raw" / "synthetic").glob("*.h5"))
        assert len(h5_files) == 8

    def test_h5_signal_shape(self, tiny_builder, tmp_path):
        tiny_builder.build()
        h5_files = list((tmp_path / "raw" / "synthetic").glob("*.h5"))
        with h5py.File(h5_files[0], "r") as f:
            signal = f["signal"][:]
            assert signal.dtype == np.float32
            assert signal.ndim == 2
            # n_channels = 1
            assert signal.shape[1] == 1

    def test_h5_metadata_present(self, tiny_builder, tmp_path):
        tiny_builder.build()
        h5_files = list((tmp_path / "raw" / "synthetic").glob("*.h5"))
        with h5py.File(h5_files[0], "r") as f:
            assert "sensor_type" in f.attrs
            assert "event_type" in f.attrs
            assert "metadata" in f
            assert "flight_id" in f["metadata"].attrs

    def test_jsonl_files_created(self, tiny_builder, tmp_path):
        tiny_builder.build()
        splits_dir = tmp_path / "splits"
        assert (splits_dir / "synthetic_train.jsonl").exists()
        assert (splits_dir / "synthetic_val.jsonl").exists()
        assert (splits_dir / "synthetic_test.jsonl").exists()

    def test_jsonl_records_valid(self, tiny_builder, tmp_path):
        tiny_builder.build()
        splits_dir = tmp_path / "splits"
        all_records = []
        for fname in ("synthetic_train.jsonl", "synthetic_val.jsonl", "synthetic_test.jsonl"):
            fpath = splits_dir / fname
            with fpath.open() as f:
                for line in f:
                    record = json.loads(line)
                    all_records.append(record)

        assert len(all_records) == 8

        required_keys = {
            "path", "sensor", "event_type", "split", "label",
            "description", "qa_pairs", "n_channels", "n_samples",
        }
        for rec in all_records:
            missing = required_keys - set(rec.keys())
            assert not missing, f"Record missing keys: {missing}"
            assert isinstance(rec["qa_pairs"], list)
            assert len(rec["qa_pairs"]) >= 3

    def test_split_assignment_complete(self, tiny_builder, tmp_path):
        records = tiny_builder.build()
        splits_seen = {r["split"] for r in records}
        # With 8 samples we might not always hit all three; test at least train
        assert "train" in splits_seen

    def test_split_ratios_error(self, tmp_path):
        with pytest.raises(ValueError, match="sum to 1.0"):
            SyntheticDatasetBuilder(
                data_root=tmp_path,
                split_ratios=(0.5, 0.3, 0.3),
            )

    def test_reproducible_build(self, tmp_path):
        """Two runs with the same seed should produce identical files."""
        small_config = SyntheticSensorConfig(
            sample_rate=512.0, duration_s=0.1, n_channels=1, rng_seed=None
        )
        builder_a = SyntheticDatasetBuilder(
            data_root=tmp_path / "run_a",
            samples_per_class=2,
            sensor_types=[SensorType.VIBRATION],
            config_overrides={SensorType.VIBRATION: small_config},
            seed=7,
        )
        builder_b = SyntheticDatasetBuilder(
            data_root=tmp_path / "run_b",
            samples_per_class=2,
            sensor_types=[SensorType.VIBRATION],
            config_overrides={SensorType.VIBRATION: small_config},
            seed=7,
        )
        records_a = builder_a.build()
        records_b = builder_b.build()
        # Descriptions should be identical
        descs_a = sorted(r["description"] for r in records_a)
        descs_b = sorted(r["description"] for r in records_b)
        assert descs_a == descs_b

    def test_all_sensor_types(self, tmp_path):
        """Builder should handle all four sensor modalities."""
        small_configs = {
            SensorType.VIBRATION: SyntheticSensorConfig(
                sample_rate=512.0, duration_s=0.1, n_channels=1, rng_seed=0
            ),
            SensorType.IMU: SyntheticSensorConfig(
                sample_rate=100.0, duration_s=0.1, n_channels=6, rng_seed=0
            ),
            SensorType.TEMPERATURE: SyntheticSensorConfig(
                sample_rate=1.0, duration_s=5.0, n_channels=1, rng_seed=0
            ),
            SensorType.PRESSURE: SyntheticSensorConfig(
                sample_rate=50.0, duration_s=0.1, n_channels=1, rng_seed=0
            ),
        }
        builder = SyntheticDatasetBuilder(
            data_root=tmp_path,
            samples_per_class=1,
            config_overrides=small_configs,
            seed=0,
        )
        records = builder.build()
        # vibration: 4, imu: 3, temperature: 3, pressure: 3 = 13 total
        assert len(records) == 13
        sensors_seen = {r["sensor"] for r in records}
        assert sensors_seen == {"vibration", "imu", "temperature", "pressure"}


# ─── CLI smoke test ───────────────────────────────────────────────────────────


class TestGenerateScriptCLI:
    """Invoke the CLI script as a subprocess to catch import/runtime errors."""

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "scripts/generate_synthetic_data.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        assert result.returncode == 0
        assert "synthetic" in result.stdout.lower()

    def test_generation_minimal(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_synthetic_data.py",
                "--data-root",
                str(tmp_path),
                "--samples",
                "2",
                "--sensors",
                "vibration",
                "--summary",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        # Verify files were created
        h5_files = list((tmp_path / "raw" / "synthetic").glob("*.h5"))
        assert len(h5_files) == 8  # 4 event types × 2 samples
