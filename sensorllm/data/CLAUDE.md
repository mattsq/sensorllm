# Data Pipeline — Agent Guide

## Purpose

This subpackage handles everything from raw sensor files on disk to PyTorch-ready
batches. Pipeline: raw sensor file → windowed time-series → dataset batch.

**No image transform step.** Sensor signals are returned as raw float32 arrays and
processed directly by the time-series encoder in the model.

## Directory Map

```
sensors/        Readers for each sensor modality (IMU, vibration, temp, pressure)
datasets/       PyTorch Dataset classes returning windowed raw signals
preprocessing/  Normalization, windowing, and augmentation utilities
```

## Sensor Data Formats

Raw sensor files expected in one of:
- **HDF5** (`.h5`): preferred for multi-channel recordings; groups by sensor type
- **CSV**: simple time-series; columns are `timestamp, ch0, ch1, ...`
- **NumPy** (`.npy` / `.npz`): pre-extracted arrays

All readers return a `SensorReading` namedtuple:
```python
SensorReading(
    signal: np.ndarray,   # shape (n_samples, n_channels)
    sample_rate: float,   # Hz
    metadata: dict,       # flight_id, sensor_id, units, etc.
)
```

## Dataset Contract

All datasets return dicts with these exact keys:

```python
{
    "sensor_signal":  torch.Tensor,  # (C, L) float32 — raw windowed signal
    "input_ids":      torch.Tensor,  # (seq_len,) long
    "attention_mask": torch.Tensor,  # (seq_len,) long
    "labels":         torch.Tensor,  # (seq_len,) long, -100 for prompt tokens
}
```

Where:
- `C` = `data.n_channels` (config) = number of sensor channels
- `L` = `data.window_size` (config) = samples per window

The model's SensorEncoder receives `sensor_signal` shaped `(B, C, L)` and encodes
it directly in the time-series domain.

## Windowing

Windowing is performed by the dataset `__getitem__` using:
- `window_size`: number of samples per window (= L)
- `hop_size`: step between window start positions (controls overlap)

Utilities in `preprocessing/windowing.py`:
- `sliding_windows(signal, window_size, hop_size)` — generator of windows
- `segment_by_event(signal, event_indices, context_before, context_after)` — event-centered windows

## Adding a New Sensor Modality

1. Create `sensors/my_sensor.py` inheriting `BaseSensorReader`
2. Implement `read(path: Path) -> SensorReading`
3. Register in `sensors/__init__.py` SENSOR_REGISTRY
4. Add unit test using a synthetic signal in `tests/unit/test_datasets.py`

## Data Split Files

Train/val/test splits are JSONL index files in `data/splits/` — **committed** to the repo.
Format: `{"path": "raw/flight_001.h5", "sensor": "vibration", "split": "train", "label": "normal"}`.
Raw data, processed arrays are gitignored.
