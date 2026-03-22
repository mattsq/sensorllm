# Data Pipeline — Agent Guide

## Purpose

This subpackage handles everything from raw sensor files on disk to PyTorch-ready
batches. Pipeline: raw sensor file → windowed time-series → image transform →
normalized tensor → dataset batch.

## Directory Map

```
sensors/        Readers for each sensor modality (IMU, vibration, temp, pressure)
transforms/     Algorithms that convert a 1D signal array into a 2D image array
datasets/       PyTorch Dataset classes wiring sensors + transforms together
preprocessing/  Normalization, windowing, and augmentation utilities
```

## Sensor Data Formats

Raw sensor files are expected in one of:
- **HDF5** (`.h5`): preferred for multi-channel recordings; groups by sensor type
- **CSV**: simple single-sensor time-series; columns are `timestamp, ch0, ch1, ...`
- **NumPy** (`.npy` / `.npz`): pre-extracted arrays

All readers must return a `SensorReading` namedtuple:
```python
SensorReading(
    signal: np.ndarray,   # shape (n_samples, n_channels)
    sample_rate: float,   # Hz
    metadata: dict,       # flight_id, sensor_id, units, etc.
)
```

## Transform Conventions

- **Input**: `np.ndarray` of shape `(n_samples,)` or `(n_samples, n_channels)`
- **Output**: `np.ndarray` of shape `(H, W)` or `(C, H, W)`, dtype `float32`, values in `[0, 1]`
- All transforms are stateless callables; parameters set at construction time
- Multi-channel signals: apply transform per channel, stack along leading C dimension
- All transforms inherit `BaseTransform` and must call `self._normalize()` on output

Registered transforms (importable from `sensorllm.data.transforms`):

| Key | Class | Description |
|-----|-------|-------------|
| `spectrogram` | `MelSpectrogramTransform` | STFT-based log mel spectrogram |
| `cwt` | `CWTTransform` | Morlet wavelet scalogram |
| `recurrence` | `RecurrencePlotTransform` | Recurrence plot (nonlinear dynamics) |
| `raw_image` | `RawImageTransform` | Reshape signal to 2D grid (no-loss baseline) |

## Dataset Contract

All datasets return dicts with these exact keys so the trainer sees a uniform interface:

```python
{
    "sensor_image":   torch.Tensor,  # (C, H, W) float32 — sensor image
    "input_ids":      torch.Tensor,  # (seq_len,) long    — tokenized text
    "attention_mask": torch.Tensor,  # (seq_len,) long    — text attention mask
    "labels":         torch.Tensor,  # (seq_len,) long    — -100 for prompt tokens
}
```

## Adding a New Sensor Modality

1. Create `sensors/my_sensor.py` with a class inheriting `BaseSensorReader`
2. Implement `read(path: Path) -> SensorReading`
3. Register in `sensors/__init__.py` SENSOR_REGISTRY dict under a string key
4. Add unit test using a synthetic signal in `tests/unit/test_datasets.py`

## Adding a New Transform

1. Create `transforms/my_transform.py` inheriting `BaseTransform`
2. Implement `__call__(signal: np.ndarray) -> np.ndarray`
3. Ensure output is float32 in [0, 1] (use `self._normalize()`)
4. Register in `transforms/__init__.py` TRANSFORM_REGISTRY dict
5. Add a visualization cell to `notebooks/02_transform_comparison.ipynb`

## Data Split Files

Train/val/test splits are maintained as index files in `data/splits/` and **are committed**
to the repo (they contain only file paths and metadata, not data). Raw data, processed
arrays, and spectrograms are gitignored.

Split files format: JSONL with fields `{"path": "relative/path.h5", "label": ..., "split": "train"}`.
