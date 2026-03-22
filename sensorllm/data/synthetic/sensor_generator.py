"""Physics-inspired synthetic sensor signal generators.

Each generator produces a raw numpy array of shape (n_samples, n_channels) with
realistic characteristics (dominant frequencies, noise floors, fault signatures)
so the full pipeline — encoder → adapter → LLM — can be exercised without real
flight data.

Supported sensor types and event types:

    vibration:    normal | bearing_fault | imbalance | misalignment
    imu:          normal | turbulence | unusual_attitude
    temperature:  normal | overheat | rapid_cooling
    pressure:     normal | pressure_loss | spike
"""

from __future__ import annotations

import dataclasses
from enum import Enum

import numpy as np


# ─── Enumerations ─────────────────────────────────────────────────────────────


class SensorType(str, Enum):
    VIBRATION = "vibration"
    IMU = "imu"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"


class EventType(str, Enum):
    # Vibration events
    NORMAL = "normal"
    BEARING_FAULT = "bearing_fault"
    IMBALANCE = "imbalance"
    MISALIGNMENT = "misalignment"
    # IMU events
    TURBULENCE = "turbulence"
    UNUSUAL_ATTITUDE = "unusual_attitude"
    # Temperature events
    OVERHEAT = "overheat"
    RAPID_COOLING = "rapid_cooling"
    # Pressure events
    PRESSURE_LOSS = "pressure_loss"
    SPIKE = "spike"


# ─── Config dataclass ──────────────────────────────────────────────────────────


@dataclasses.dataclass
class SyntheticSensorConfig:
    """Parameters controlling the synthetic signal generation.

    Attributes:
        sample_rate: Sampling frequency in Hz.
        duration_s: Signal duration in seconds.
        n_channels: Number of sensor channels.
        noise_std: Standard deviation of additive Gaussian noise.
        rng_seed: Optional seed for reproducibility (None = random).
    """

    sample_rate: float = 4096.0
    duration_s: float = 1.0
    n_channels: int = 1
    noise_std: float = 0.05
    rng_seed: int | None = None

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate * self.duration_s)


# ─── Default configs per sensor type ──────────────────────────────────────────

_VIBRATION_CFG = SyntheticSensorConfig(sample_rate=4096.0, duration_s=1.0, n_channels=1, noise_std=0.05)
_IMU_CFG = SyntheticSensorConfig(sample_rate=400.0, duration_s=1.0, n_channels=6, noise_std=0.01)
_TEMPERATURE_CFG = SyntheticSensorConfig(sample_rate=1.0, duration_s=60.0, n_channels=1, noise_std=0.5)
_PRESSURE_CFG = SyntheticSensorConfig(sample_rate=100.0, duration_s=1.0, n_channels=1, noise_std=0.02)


# ─── Vibration generators ──────────────────────────────────────────────────────


def generate_vibration_signal(
    event_type: EventType = EventType.NORMAL,
    config: SyntheticSensorConfig | None = None,
    rotation_hz: float = 50.0,
) -> np.ndarray:
    """Generate a synthetic vibration/accelerometer signal.

    Args:
        event_type: The operating condition or fault type to simulate.
        config: Signal generation parameters. Defaults to 4096 Hz, 1 s, 1 channel.
        rotation_hz: Shaft rotation frequency in Hz (basis for fault harmonics).

    Returns:
        Signal array of shape (n_samples, n_channels), dtype float32.
    """
    if config is None:
        config = _VIBRATION_CFG
    rng = np.random.default_rng(config.rng_seed)
    t = np.linspace(0.0, config.duration_s, config.n_samples, endpoint=False, dtype=np.float64)

    channels = []
    for ch in range(config.n_channels):
        phase_offset = ch * np.pi / 4

        # ── Normal baseline: fundamental + 2nd harmonic ──────────────────────
        sig = (
            np.sin(2 * np.pi * rotation_hz * t + phase_offset)
            + 0.3 * np.sin(2 * np.pi * 2 * rotation_hz * t + phase_offset)
            + config.noise_std * rng.standard_normal(config.n_samples)
        )

        if event_type == EventType.BEARING_FAULT:
            # Ball-pass outer-race frequency (BPFO) ≈ 3.585 × rotation_hz
            bpfo = 3.585 * rotation_hz
            # Amplitude-modulated impulse train at BPFO with random phase jitter
            impulse_times = np.arange(0, config.duration_s, 1.0 / bpfo)
            for t0 in impulse_times:
                t0 += rng.uniform(-0.5 / bpfo, 0.5 / bpfo)  # jitter
                idx = int(t0 * config.sample_rate)
                if 0 <= idx < config.n_samples - 20:
                    # Decaying exponential impulse
                    decay = np.exp(-500 * (t[idx : idx + 20] - t[idx]))
                    sig[idx : idx + 20] += 0.8 * decay
            # Add modulation sidebands
            sig += 0.2 * np.sin(2 * np.pi * bpfo * t) * np.sin(2 * np.pi * rotation_hz * t)

        elif event_type == EventType.IMBALANCE:
            # Imbalance: dominant 1× component with elevated amplitude
            sig += 1.5 * np.sin(2 * np.pi * rotation_hz * t + phase_offset + 0.3)

        elif event_type == EventType.MISALIGNMENT:
            # Misalignment: strong 2× and 3× components
            sig += (
                1.2 * np.sin(2 * np.pi * 2 * rotation_hz * t + phase_offset)
                + 0.6 * np.sin(2 * np.pi * 3 * rotation_hz * t + phase_offset)
            )

        channels.append(sig.astype(np.float32))

    return np.stack(channels, axis=1)  # (n_samples, n_channels)


# ─── IMU generators ───────────────────────────────────────────────────────────


def generate_imu_signal(
    event_type: EventType = EventType.NORMAL,
    config: SyntheticSensorConfig | None = None,
) -> np.ndarray:
    """Generate a synthetic IMU signal (3-axis gyroscope + 3-axis accelerometer).

    Channel layout: [gx, gy, gz, ax, ay, az]
    - Gyroscope: degrees/second, small baseline oscillation
    - Accelerometer: m/s², gravity in az + small vibration

    Args:
        event_type: The operating condition to simulate.
        config: Signal generation parameters. Defaults to 400 Hz, 1 s, 6 channels.

    Returns:
        Signal array of shape (n_samples, 6), dtype float32.
    """
    if config is None:
        config = _IMU_CFG
    rng = np.random.default_rng(config.rng_seed)
    t = np.linspace(0.0, config.duration_s, config.n_samples, endpoint=False, dtype=np.float64)

    # Gyroscope: low-freq oscillation + noise (deg/s)
    gx = 0.5 * np.sin(2 * np.pi * 0.5 * t) + config.noise_std * rng.standard_normal(config.n_samples)
    gy = 0.3 * np.sin(2 * np.pi * 0.3 * t + 0.5) + config.noise_std * rng.standard_normal(config.n_samples)
    gz = 0.2 * np.sin(2 * np.pi * 0.2 * t + 1.0) + config.noise_std * rng.standard_normal(config.n_samples)

    # Accelerometer: gravity in z + small vibration (m/s²)
    ax = 0.1 * np.sin(2 * np.pi * 2.0 * t) + config.noise_std * rng.standard_normal(config.n_samples)
    ay = 0.1 * np.sin(2 * np.pi * 1.5 * t + 0.8) + config.noise_std * rng.standard_normal(config.n_samples)
    az = 9.81 + 0.05 * np.sin(2 * np.pi * 3.0 * t) + config.noise_std * rng.standard_normal(config.n_samples)

    if event_type == EventType.TURBULENCE:
        # Turbulence: broadband high-amplitude noise bursts
        burst_std = 5.0
        gx += burst_std * rng.standard_normal(config.n_samples) * _random_envelope(t, rng)
        gy += burst_std * rng.standard_normal(config.n_samples) * _random_envelope(t, rng)
        gz += burst_std * rng.standard_normal(config.n_samples) * _random_envelope(t, rng)
        ax += burst_std * rng.standard_normal(config.n_samples) * _random_envelope(t, rng)
        ay += burst_std * rng.standard_normal(config.n_samples) * _random_envelope(t, rng)
        az += burst_std * rng.standard_normal(config.n_samples) * _random_envelope(t, rng)

    elif event_type == EventType.UNUSUAL_ATTITUDE:
        # Unusual attitude: sustained roll/pitch offset
        roll_rate = 15.0  # deg/s sustained roll
        gx += roll_rate * np.ones(config.n_samples)
        # Gravity vector shifts from z to x (approx 30° bank)
        ax += 9.81 * np.sin(np.deg2rad(30)) * np.ones(config.n_samples)
        az = 9.81 * np.cos(np.deg2rad(30)) * np.ones(config.n_samples) + 0.05 * rng.standard_normal(config.n_samples)

    channels = np.stack([gx, gy, gz, ax, ay, az], axis=1).astype(np.float32)
    return channels  # (n_samples, 6)


def _random_envelope(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Smooth random amplitude envelope in [0, 1] for burst simulation."""
    n = len(t)
    # Low-frequency random walk, smoothed
    raw = np.abs(rng.standard_normal(n))
    # Smooth with a simple uniform filter
    kernel_size = max(1, n // 20)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(raw, kernel, mode="same")
    vmin, vmax = smoothed.min(), smoothed.max()
    if vmax > vmin:
        smoothed = (smoothed - vmin) / (vmax - vmin)
    return smoothed.astype(np.float64)


# ─── Temperature generators ───────────────────────────────────────────────────


def generate_temperature_signal(
    event_type: EventType = EventType.NORMAL,
    config: SyntheticSensorConfig | None = None,
    nominal_temp_c: float = 80.0,
) -> np.ndarray:
    """Generate a synthetic engine temperature signal.

    Args:
        event_type: The operating condition to simulate.
        config: Signal generation parameters. Defaults to 1 Hz, 60 s, 1 channel.
        nominal_temp_c: Nominal operating temperature in degrees Celsius.

    Returns:
        Signal array of shape (n_samples, n_channels), dtype float32.
    """
    if config is None:
        config = _TEMPERATURE_CFG
    rng = np.random.default_rng(config.rng_seed)
    t = np.linspace(0.0, config.duration_s, config.n_samples, endpoint=False, dtype=np.float64)

    channels = []
    for ch in range(config.n_channels):
        # Normal: slow sinusoidal fluctuation around nominal
        sig = (
            nominal_temp_c
            + 2.0 * np.sin(2 * np.pi * (1.0 / config.duration_s) * t)
            + config.noise_std * rng.standard_normal(config.n_samples)
        )

        if event_type == EventType.OVERHEAT:
            # Monotonically increasing temperature exceeding limit (e.g., > 120 °C)
            overheat_rate = (150.0 - nominal_temp_c) / config.duration_s
            sig += overheat_rate * t

        elif event_type == EventType.RAPID_COOLING:
            # Sharp negative ramp — sudden cooling (e.g., fuel shutoff)
            onset = int(0.3 * config.n_samples)
            ramp = np.zeros(config.n_samples)
            ramp[onset:] = -60.0 * (t[onset:] - t[onset]) / (config.duration_s - t[onset])
            sig += ramp

        channels.append(sig.astype(np.float32))

    return np.stack(channels, axis=1)  # (n_samples, n_channels)


# ─── Pressure generators ──────────────────────────────────────────────────────


def generate_pressure_signal(
    event_type: EventType = EventType.NORMAL,
    config: SyntheticSensorConfig | None = None,
    nominal_psi: float = 14.7,
) -> np.ndarray:
    """Generate a synthetic pressure sensor signal.

    Args:
        event_type: The operating condition to simulate.
        config: Signal generation parameters. Defaults to 100 Hz, 1 s, 1 channel.
        nominal_psi: Nominal operating pressure in PSI.

    Returns:
        Signal array of shape (n_samples, n_channels), dtype float32.
    """
    if config is None:
        config = _PRESSURE_CFG
    rng = np.random.default_rng(config.rng_seed)
    t = np.linspace(0.0, config.duration_s, config.n_samples, endpoint=False, dtype=np.float64)

    channels = []
    for ch in range(config.n_channels):
        # Normal: stable with minor fluctuations
        sig = (
            nominal_psi
            + 0.1 * np.sin(2 * np.pi * 5.0 * t)
            + config.noise_std * rng.standard_normal(config.n_samples)
        )

        if event_type == EventType.PRESSURE_LOSS:
            # Gradual pressure loss — linear decline
            loss_rate = nominal_psi * 0.8 / config.duration_s
            sig -= loss_rate * t

        elif event_type == EventType.SPIKE:
            # Brief pressure spike at random location.
            # np.hanning requires N >= 3 (N=1 → [1.0], N=2 → [0., 0.]).
            spike_idx = rng.integers(config.n_samples // 4, 3 * config.n_samples // 4)
            spike_width = max(3, int(0.02 * config.sample_rate))  # ~20 ms, min 3 samples
            spike_envelope = np.zeros(config.n_samples)
            half = spike_width // 2
            start = max(0, spike_idx - half)
            end = min(config.n_samples, start + spike_width)
            window = np.hanning(end - start)
            spike_envelope[start:end] = window
            sig += 5.0 * spike_envelope  # +5 PSI spike

        channels.append(sig.astype(np.float32))

    return np.stack(channels, axis=1)  # (n_samples, n_channels)


# ─── Dispatch table ───────────────────────────────────────────────────────────

#: Maps sensor type → set of valid event types for that sensor.
VALID_EVENTS: dict[SensorType, list[EventType]] = {
    SensorType.VIBRATION: [
        EventType.NORMAL,
        EventType.BEARING_FAULT,
        EventType.IMBALANCE,
        EventType.MISALIGNMENT,
    ],
    SensorType.IMU: [
        EventType.NORMAL,
        EventType.TURBULENCE,
        EventType.UNUSUAL_ATTITUDE,
    ],
    SensorType.TEMPERATURE: [
        EventType.NORMAL,
        EventType.OVERHEAT,
        EventType.RAPID_COOLING,
    ],
    SensorType.PRESSURE: [
        EventType.NORMAL,
        EventType.PRESSURE_LOSS,
        EventType.SPIKE,
    ],
}

#: Default configs per sensor type.
DEFAULT_CONFIGS: dict[SensorType, SyntheticSensorConfig] = {
    SensorType.VIBRATION: _VIBRATION_CFG,
    SensorType.IMU: _IMU_CFG,
    SensorType.TEMPERATURE: _TEMPERATURE_CFG,
    SensorType.PRESSURE: _PRESSURE_CFG,
}


def generate_signal(
    sensor_type: SensorType,
    event_type: EventType,
    config: SyntheticSensorConfig | None = None,
) -> np.ndarray:
    """Dispatch to the appropriate sensor generator.

    Args:
        sensor_type: Which physical sensor to simulate.
        event_type: Operating condition or fault scenario.
        config: Optional config override; falls back to sensor-type defaults.

    Returns:
        Signal array of shape (n_samples, n_channels), dtype float32.

    Raises:
        ValueError: If event_type is not valid for the given sensor_type.
    """
    if event_type not in VALID_EVENTS[sensor_type]:
        raise ValueError(
            f"Event '{event_type}' is not valid for sensor '{sensor_type}'. "
            f"Valid options: {[e.value for e in VALID_EVENTS[sensor_type]]}"
        )

    generators = {
        SensorType.VIBRATION: generate_vibration_signal,
        SensorType.IMU: generate_imu_signal,
        SensorType.TEMPERATURE: generate_temperature_signal,
        SensorType.PRESSURE: generate_pressure_signal,
    }
    return generators[sensor_type](event_type=event_type, config=config)
