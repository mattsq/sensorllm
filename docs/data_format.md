# Sensor Data Format Specification

## Supported Raw File Formats

### HDF5 (`.h5`) — Preferred

Preferred format for multi-channel, multi-sensor recordings.

Expected structure:
```
flight_001.h5
├── imu/
│   ├── accel_x     Dataset: (n_samples,) float32
│   ├── accel_y     Dataset: (n_samples,) float32
│   ├── accel_z     Dataset: (n_samples,) float32
│   ├── gyro_x      Dataset: (n_samples,) float32
│   ├── gyro_y      Dataset: (n_samples,) float32
│   └── gyro_z      Dataset: (n_samples,) float32
├── vibration/
│   └── channel_0   Dataset: (n_samples,) float32
├── temperature/
│   ├── egt         Dataset: (n_samples,) float32 (exhaust gas temp)
│   └── cht         Dataset: (n_samples,) float32 (cylinder head temp)
└── attrs:
    flight_id       str
    sample_rates    dict  {sensor_group: float_Hz}
    units           dict  {channel: unit_string}
    start_time      str   ISO 8601 UTC
```

### CSV (`.csv`)

Simple format for single-sensor time-series:
```
timestamp,channel_0,channel_1,...
0.000,1.234,-0.567,...
0.001,1.240,-0.570,...
```
- First column must be `timestamp` (seconds from start, float)
- Remaining columns are sensor channels
- File-level metadata (sample_rate, sensor_type) stored in companion `.json`

### NumPy (`.npy` / `.npz`)

Pre-extracted arrays. `.npz` files should contain:
- `signal`: `(n_samples, n_channels)` float32
- `sample_rate`: scalar
- `metadata`: pickled dict (via `np.savez(..., metadata=json.dumps(meta))`)

## SensorReading Output Contract

All readers return:
```python
SensorReading(
    signal=np.ndarray,      # shape (n_samples, n_channels), float32
    sample_rate=float,      # Hz
    metadata=dict,          # flight_id, sensor_id, units, anomaly_labels, etc.
)
```

## Signal Units

| Sensor | Typical Unit | Typical Sample Rate |
|--------|-------------|-------------------|
| IMU Accelerometer | g (gravitational units) or m/s² | 400 Hz |
| IMU Gyroscope | deg/s or rad/s | 400 Hz |
| Vibration | g | 1–20 kHz |
| Temperature | °C | 1–10 Hz |
| Pressure | hPa or PSI | 10–100 Hz |

## Annotation Format

Labels/annotations are stored in split JSONL files (`data/splits/*.jsonl`):
```json
{
    "path": "raw/flight_001.h5",
    "sensor": "vibration",
    "split": "train",
    "label": "normal",
    "anomaly_type": null,
    "qa_pairs": [
        {"question": "Is there any anomaly?", "answer": "No anomaly detected."},
        {"question": "Describe the vibration signature.", "answer": "Normal sinusoidal pattern at 50 Hz."}
    ]
}
```
