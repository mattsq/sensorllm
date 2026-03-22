"""Template-based annotation generator for synthetic sensor data.

Produces two annotation types for each (sensor_type, event_type) sample:

    1. **Pretrain description** — a short, factual description of the signal
       characteristics used in Stage 1 adapter alignment.

    2. **QA pairs** — a list of question-answer dicts covering anomaly detection,
       fault diagnosis, and operational state narration used in Stage 2
       instruction fine-tuning.

All text is deterministic given the event type, with optional numeric signal
statistics (RMS, peak, dominant frequency) interpolated in.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from sensorllm.data.synthetic.sensor_generator import EventType, SensorType


# ─── Signal statistics ────────────────────────────────────────────────────────


@dataclasses.dataclass
class SignalStats:
    """Scalar statistics derived from a raw sensor signal window.

    Attributes:
        rms: Root-mean-square amplitude.
        peak: Peak absolute amplitude.
        mean: Mean value across all samples and channels.
        std: Standard deviation across all samples and channels.
        dominant_freq_hz: Frequency of the highest-amplitude FFT bin (first channel).
    """

    rms: float
    peak: float
    mean: float
    std: float
    dominant_freq_hz: float | None


def compute_stats(signal: np.ndarray, sample_rate: float) -> SignalStats:
    """Compute scalar statistics from a raw sensor signal.

    Args:
        signal: Shape (n_samples, n_channels), dtype float32.
        sample_rate: Sampling frequency in Hz.

    Returns:
        SignalStats with rms, peak, mean, std, dominant_freq_hz.
    """
    flat = signal.flatten().astype(np.float64)
    rms = float(np.sqrt(np.mean(flat**2)))
    peak = float(np.max(np.abs(flat)))
    mean = float(np.mean(flat))
    std = float(np.std(flat))

    # Dominant frequency from first channel's FFT
    first_ch = signal[:, 0].astype(np.float64)
    n = len(first_ch)
    fft_mag = np.abs(np.fft.rfft(first_ch * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    # Ignore DC component (bin 0)
    dominant_freq_hz = float(freqs[1 + np.argmax(fft_mag[1:])]) if n > 1 else None

    return SignalStats(rms=rms, peak=peak, mean=mean, std=std, dominant_freq_hz=dominant_freq_hz)


# ─── Annotation templates ──────────────────────────────────────────────────────

_PRETRAIN_TEMPLATES: dict[tuple[SensorType, EventType], str] = {
    # ── Vibration ──────────────────────────────────────────────────────────────
    (SensorType.VIBRATION, EventType.NORMAL): (
        "The vibration sensor shows a normal operating profile. "
        "The dominant frequency is {dominant_freq_hz:.1f} Hz with RMS amplitude {rms:.3f} g. "
        "No anomalous spectral components are present."
    ),
    (SensorType.VIBRATION, EventType.BEARING_FAULT): (
        "The vibration sensor exhibits a bearing fault signature. "
        "Periodic impulses at the ball-pass outer-race frequency (BPFO ≈ {dominant_freq_hz:.1f} Hz) "
        "are superimposed on the normal rotation harmonics. "
        "Peak amplitude is {peak:.3f} g, elevated above the normal baseline."
    ),
    (SensorType.VIBRATION, EventType.IMBALANCE): (
        "The vibration sensor indicates a mass imbalance condition. "
        "The fundamental rotation frequency at {dominant_freq_hz:.1f} Hz is significantly "
        "elevated (RMS {rms:.3f} g), with minimal higher-harmonic content — "
        "characteristic of a rotating imbalance fault."
    ),
    (SensorType.VIBRATION, EventType.MISALIGNMENT): (
        "The vibration sensor shows a shaft misalignment pattern. "
        "The second and third harmonics of the rotation frequency are dominant, "
        "consistent with angular or parallel misalignment. "
        "Peak amplitude is {peak:.3f} g."
    ),
    # ── IMU ───────────────────────────────────────────────────────────────────
    (SensorType.IMU, EventType.NORMAL): (
        "The IMU reading reflects normal flight dynamics. "
        "Gyroscope rates are within ±2 deg/s and accelerometer axes show "
        "the expected gravity vector with minor vibration (RMS {rms:.3f} m/s² or deg/s)."
    ),
    (SensorType.IMU, EventType.TURBULENCE): (
        "The IMU data indicates an episode of atmospheric turbulence. "
        "All six axes show irregular, broadband oscillations with RMS {rms:.3f} "
        "and peak excursion of {peak:.3f}, significantly above normal levels."
    ),
    (SensorType.IMU, EventType.UNUSUAL_ATTITUDE): (
        "The IMU data reveals an unusual aircraft attitude. "
        "Sustained roll rate of approximately 15 deg/s is observed on the x-gyroscope, "
        "and the accelerometer z-axis no longer aligns with the expected gravity vector, "
        "indicating a bank angle of roughly 30 degrees."
    ),
    # ── Temperature ───────────────────────────────────────────────────────────
    (SensorType.TEMPERATURE, EventType.NORMAL): (
        "The temperature sensor shows normal engine operating conditions. "
        "Temperature is stable around {mean:.1f} °C with minor cyclic variation "
        "(std {std:.2f} °C), within the operational envelope."
    ),
    (SensorType.TEMPERATURE, EventType.OVERHEAT): (
        "The temperature sensor has detected an overheating event. "
        "Temperature is rising monotonically, reaching a peak of {peak:.1f} °C, "
        "which significantly exceeds the normal operating limit of 120 °C. "
        "Immediate attention is required."
    ),
    (SensorType.TEMPERATURE, EventType.RAPID_COOLING): (
        "The temperature sensor shows a rapid cooling event. "
        "After an initial normal reading near {mean:.1f} °C, a sharp temperature "
        "decrease is observed, potentially indicating fuel shutoff or coolant loss."
    ),
    # ── Pressure ──────────────────────────────────────────────────────────────
    (SensorType.PRESSURE, EventType.NORMAL): (
        "The pressure sensor indicates normal operating conditions. "
        "Pressure is stable at approximately {mean:.2f} PSI with minor fluctuations "
        "(peak deviation {peak:.3f} PSI), consistent with routine operation."
    ),
    (SensorType.PRESSURE, EventType.PRESSURE_LOSS): (
        "The pressure sensor has detected a gradual pressure loss. "
        "Pressure has declined from the nominal {mean:.2f} PSI by more than 80%, "
        "indicating a significant leak or seal failure requiring immediate diagnosis."
    ),
    (SensorType.PRESSURE, EventType.SPIKE): (
        "The pressure sensor recorded a transient pressure spike. "
        "A brief excursion of approximately {peak:.2f} PSI above the nominal operating "
        "pressure was observed, lasting roughly 20 ms — consistent with a hydraulic "
        "hammer event or valve transient."
    ),
}

_QA_TEMPLATES: dict[tuple[SensorType, EventType], list[dict[str, str]]] = {
    # ── Vibration ──────────────────────────────────────────────────────────────
    (SensorType.VIBRATION, EventType.NORMAL): [
        {
            "question": "Is there an anomaly present in this vibration sensor reading?",
            "answer": "No. The vibration reading shows a normal profile with the expected "
            "rotation harmonics and a low noise floor. No anomalous spectral components "
            "are detected.",
        },
        {
            "question": "What is the dominant frequency component in this vibration signal?",
            "answer": "The dominant frequency is the shaft rotation fundamental at "
            "approximately {dominant_freq_hz:.1f} Hz, with a minor second harmonic — "
            "consistent with normal rotating machinery.",
        },
        {
            "question": "What is the operational state of the component based on this reading?",
            "answer": "The component is operating normally. Vibration amplitude is within "
            "acceptable limits (RMS {rms:.3f} g) and the spectral content shows no "
            "fault indicators.",
        },
    ],
    (SensorType.VIBRATION, EventType.BEARING_FAULT): [
        {
            "question": "Is there an anomaly present in this vibration sensor reading?",
            "answer": "Yes. The signal contains periodic impulses at the ball-pass "
            "outer-race frequency (BPFO), which is a hallmark of an outer-race "
            "bearing defect. The peak amplitude is {peak:.3f} g.",
        },
        {
            "question": "What type of fault does this vibration reading indicate?",
            "answer": "The reading indicates a bearing fault, specifically an outer-race "
            "defect. The BPFO-spaced impulse train superimposed on the rotation "
            "harmonics is the characteristic signature.",
        },
        {
            "question": "What corrective action is recommended based on this reading?",
            "answer": "The affected bearing should be scheduled for inspection or "
            "replacement at the earliest opportunity. Continued operation risks "
            "secondary damage to the shaft and housing.",
        },
    ],
    (SensorType.VIBRATION, EventType.IMBALANCE): [
        {
            "question": "Is there an anomaly present in this vibration sensor reading?",
            "answer": "Yes. The signal shows an abnormally elevated 1× rotation frequency "
            "component (RMS {rms:.3f} g), with minimal higher harmonics — "
            "the classic signature of mass imbalance.",
        },
        {
            "question": "What type of fault does this vibration reading indicate?",
            "answer": "The reading indicates a rotating mass imbalance. The dominant 1× "
            "component at {dominant_freq_hz:.1f} Hz is significantly above the "
            "normal baseline.",
        },
        {
            "question": "What corrective action is recommended for this fault?",
            "answer": "Dynamic balancing of the rotating assembly should be performed. "
            "The rotor should be inspected for missing balance weights, loose "
            "material, or uneven wear.",
        },
    ],
    (SensorType.VIBRATION, EventType.MISALIGNMENT): [
        {
            "question": "Is there an anomaly present in this vibration sensor reading?",
            "answer": "Yes. The vibration spectrum shows elevated 2× and 3× rotation "
            "frequency components (peak {peak:.3f} g), indicative of shaft "
            "misalignment.",
        },
        {
            "question": "What type of fault does this vibration reading indicate?",
            "answer": "The reading indicates angular or parallel shaft misalignment. "
            "The strong second and third harmonics relative to the fundamental "
            "are the distinguishing signatures.",
        },
        {
            "question": "How urgent is the corrective action required?",
            "answer": "Misalignment should be corrected promptly. Left unaddressed, it "
            "accelerates bearing and coupling wear and can lead to shaft fatigue "
            "failure over time.",
        },
    ],
    # ── IMU ───────────────────────────────────────────────────────────────────
    (SensorType.IMU, EventType.NORMAL): [
        {
            "question": "Does the IMU data indicate any flight anomaly?",
            "answer": "No. The IMU data is within normal limits. Gyroscope rates are "
            "below ±2 deg/s and the accelerometer z-axis correctly shows the "
            "gravity vector. Flight dynamics appear normal.",
        },
        {
            "question": "What is the aircraft attitude indicated by this IMU reading?",
            "answer": "The aircraft is in a wings-level attitude with no significant "
            "roll, pitch, or yaw rates. The gravity vector is aligned with the "
            "vertical axis as expected.",
        },
        {
            "question": "What is the operational state based on the IMU data?",
            "answer": "Normal cruise or stable flight. All inertial parameters are "
            "within operational limits.",
        },
    ],
    (SensorType.IMU, EventType.TURBULENCE): [
        {
            "question": "Does the IMU data indicate any flight anomaly?",
            "answer": "Yes. All six IMU axes show irregular, high-amplitude oscillations "
            "(RMS {rms:.3f}, peak {peak:.3f}) characteristic of moderate to severe "
            "atmospheric turbulence.",
        },
        {
            "question": "How severe is the turbulence indicated by this IMU reading?",
            "answer": "The turbulence appears moderate to severe based on the broadband "
            "excitations across all axes. Passenger safety protocols and flight crew "
            "advisories should be activated.",
        },
        {
            "question": "What action should the flight crew take given this IMU data?",
            "answer": "The crew should reduce airspeed to the recommended turbulence "
            "penetration speed, activate the fasten-seatbelt sign, and request a "
            "ride report from ATC to identify smoother altitudes.",
        },
    ],
    (SensorType.IMU, EventType.UNUSUAL_ATTITUDE): [
        {
            "question": "Does the IMU data indicate any flight anomaly?",
            "answer": "Yes. The IMU data shows a sustained roll rate of approximately "
            "15 deg/s and a gravity-vector shift consistent with a 30-degree bank "
            "angle — an unusual attitude requiring immediate attention.",
        },
        {
            "question": "What is the aircraft attitude indicated by this IMU reading?",
            "answer": "The aircraft is in an unintended bank of approximately 30 degrees "
            "with an ongoing roll rate. If uncorrected, the bank angle will continue "
            "to increase.",
        },
        {
            "question": "What is the urgency of this situation based on the IMU data?",
            "answer": "This is an immediate safety concern. An unusual attitude recovery "
            "procedure should be executed: level the wings, verify attitude on "
            "primary and standby instruments, and check for autopilot disengagement.",
        },
    ],
    # ── Temperature ───────────────────────────────────────────────────────────
    (SensorType.TEMPERATURE, EventType.NORMAL): [
        {
            "question": "Is the engine temperature within normal operating limits?",
            "answer": "Yes. The temperature is stable around {mean:.1f} °C with minor "
            "cyclic variation, well within the normal operating envelope.",
        },
        {
            "question": "Does this temperature reading require any crew action?",
            "answer": "No action required. The temperature reading is nominal and shows "
            "no trend toward overheating or unexpected cooling.",
        },
        {
            "question": "What is the thermal state of the engine based on this reading?",
            "answer": "The engine is at normal operating temperature ({mean:.1f} °C). "
            "Thermal management systems are functioning correctly.",
        },
    ],
    (SensorType.TEMPERATURE, EventType.OVERHEAT): [
        {
            "question": "Is the engine temperature within normal operating limits?",
            "answer": "No. The temperature has risen to {peak:.1f} °C, significantly "
            "above the normal limit of 120 °C. An overheating event is in progress.",
        },
        {
            "question": "What action is required based on this temperature reading?",
            "answer": "Immediate action is required. Engine power should be reduced, "
            "cooling system status verified, and if temperature does not stabilize, "
            "engine shutdown should be considered per the abnormal procedures checklist.",
        },
        {
            "question": "What is the likely cause of this temperature anomaly?",
            "answer": "Possible causes include cooling system failure (blocked coolant "
            "flow, low coolant level), oil system malfunction, or an internal engine "
            "fault generating excessive heat.",
        },
    ],
    (SensorType.TEMPERATURE, EventType.RAPID_COOLING): [
        {
            "question": "Is there an anomaly in this temperature reading?",
            "answer": "Yes. A rapid temperature decrease is observed after the initial "
            "normal reading, which is inconsistent with normal engine operation. "
            "This may indicate fuel shutoff or coolant loss.",
        },
        {
            "question": "What could cause the rapid temperature drop seen in this data?",
            "answer": "Rapid cooling can result from fuel flow interruption (engine "
            "flame-out), an open bleed air valve, coolant line rupture, or "
            "inadvertent engine shutdown.",
        },
        {
            "question": "What is the recommended crew response to this temperature reading?",
            "answer": "Verify engine running parameters, check fuel flow and fuel "
            "quantity indicators, review ECAM/EICAS messages, and follow the "
            "relevant abnormal or emergency procedure.",
        },
    ],
    # ── Pressure ──────────────────────────────────────────────────────────────
    (SensorType.PRESSURE, EventType.NORMAL): [
        {
            "question": "Is the system pressure within normal operating limits?",
            "answer": "Yes. Pressure is stable at {mean:.2f} PSI with minor fluctuations, "
            "consistent with normal system operation.",
        },
        {
            "question": "Does this pressure reading indicate any fault?",
            "answer": "No faults detected. The pressure reading is nominal with no "
            "trending anomalies.",
        },
        {
            "question": "What is the operational state of the hydraulic/pneumatic system?",
            "answer": "The system is operating normally at {mean:.2f} PSI. No leaks, "
            "blockages, or valve faults are indicated.",
        },
    ],
    (SensorType.PRESSURE, EventType.PRESSURE_LOSS): [
        {
            "question": "Is there an anomaly in this pressure reading?",
            "answer": "Yes. Pressure has declined by more than 80% from nominal, "
            "indicating a significant system pressure loss — likely due to a "
            "leak, seal failure, or pump malfunction.",
        },
        {
            "question": "How urgent is the pressure loss indicated by this reading?",
            "answer": "This is an urgent situation. Rapid pressure loss can render "
            "hydraulic flight controls, braking, or other critical systems "
            "inoperative. Immediate crew notification and checklist action are required.",
        },
        {
            "question": "What are the likely root causes of this pressure anomaly?",
            "answer": "Likely causes include hydraulic line rupture, seal failure, "
            "reservoir depletion, or pump failure. The affected system should be "
            "isolated if possible and the backup system activated.",
        },
    ],
    (SensorType.PRESSURE, EventType.SPIKE): [
        {
            "question": "Is there an anomaly in this pressure reading?",
            "answer": "Yes. A brief pressure spike of approximately {peak:.2f} PSI "
            "above nominal was recorded, lasting roughly 20 ms — indicative of "
            "a hydraulic hammer event or valve transient.",
        },
        {
            "question": "Is this pressure spike a safety concern?",
            "answer": "A single brief spike of this magnitude is typically a minor "
            "event, but repeated spikes can indicate valve wear or water hammer "
            "issues that should be investigated during the next maintenance interval.",
        },
        {
            "question": "What is the likely cause of the pressure spike?",
            "answer": "The spike is consistent with a hydraulic hammer caused by "
            "rapid valve closure, or a momentary load transient in the system. "
            "Valve condition and line routing should be reviewed.",
        },
    ],
}


# ─── Annotation generator ──────────────────────────────────────────────────────


class AnnotationGenerator:
    """Generates natural-language annotations for synthetic sensor samples.

    Usage::

        gen = AnnotationGenerator()
        desc = gen.pretrain_description(SensorType.VIBRATION, EventType.BEARING_FAULT, signal, sr)
        pairs = gen.qa_pairs(SensorType.VIBRATION, EventType.BEARING_FAULT, signal, sr)
    """

    def pretrain_description(
        self,
        sensor_type: SensorType,
        event_type: EventType,
        signal: np.ndarray,
        sample_rate: float,
    ) -> str:
        """Return a short factual description of the signal for Stage 1 pretraining.

        Args:
            sensor_type: Physical sensor type.
            event_type: Operating condition / fault type.
            signal: Raw signal array (n_samples, n_channels).
            sample_rate: Sampling frequency in Hz.

        Returns:
            A natural language description string with statistics interpolated.
        """
        stats = compute_stats(signal, sample_rate)
        template = _PRETRAIN_TEMPLATES[(sensor_type, event_type)]
        return template.format(
            rms=stats.rms,
            peak=stats.peak,
            mean=stats.mean,
            std=stats.std,
            dominant_freq_hz=stats.dominant_freq_hz if stats.dominant_freq_hz is not None else 0.0,
        )

    def qa_pairs(
        self,
        sensor_type: SensorType,
        event_type: EventType,
        signal: np.ndarray,
        sample_rate: float,
    ) -> list[dict[str, str]]:
        """Return a list of question-answer dicts for Stage 2 instruction fine-tuning.

        Each dict has keys ``question`` and ``answer``.

        Args:
            sensor_type: Physical sensor type.
            event_type: Operating condition / fault type.
            signal: Raw signal array (n_samples, n_channels).
            sample_rate: Sampling frequency in Hz.

        Returns:
            List of {"question": ..., "answer": ...} dicts.
        """
        stats = compute_stats(signal, sample_rate)
        fmt_kwargs: dict[str, Any] = dict(
            rms=stats.rms,
            peak=stats.peak,
            mean=stats.mean,
            std=stats.std,
            dominant_freq_hz=stats.dominant_freq_hz if stats.dominant_freq_hz is not None else 0.0,
        )
        pairs = []
        for raw in _QA_TEMPLATES[(sensor_type, event_type)]:
            pairs.append(
                {
                    "question": raw["question"].format(**fmt_kwargs),
                    "answer": raw["answer"].format(**fmt_kwargs),
                }
            )
        return pairs

    def format_pretrain_prompt(self, description: str) -> tuple[str, str]:
        """Format a pretrain description as an (instruction, response) pair.

        Args:
            description: Raw description string.

        Returns:
            Tuple of (instruction_text, response_text).
        """
        instruction = (
            "You are analyzing aircraft sensor data. "
            "Describe the sensor reading in one or two sentences."
        )
        return instruction, description

    def format_qa_prompt(self, question: str, answer: str) -> tuple[str, str]:
        """Format a QA pair as an (instruction, response) pair.

        Args:
            question: Question string.
            answer: Answer string.

        Returns:
            Tuple of (instruction_text, response_text).
        """
        instruction = (
            "You are an expert aircraft systems analyst. "
            "Answer the following question about the sensor data: " + question
        )
        return instruction, answer
