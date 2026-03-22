# Data Splits

This directory contains train/val/test split index files. These **are committed** to
the repository — they contain only file paths and metadata, not raw sensor data.

## File Format

Split files are JSONL (one JSON object per line):

```json
{"path": "raw/flight_001.h5", "sensor": "vibration", "split": "train", "label": "normal", "flight_id": "F001"}
{"path": "raw/flight_002.h5", "sensor": "vibration", "split": "val",   "label": "anomaly", "flight_id": "F002"}
```

## Split Methodology

Document the split methodology here when real data is available:
- How train/val/test splits were determined (e.g., by flight date, random stratified, etc.)
- Class balance per split
- Any data leakage considerations (e.g., no flight ID appears in multiple splits)
- Version/hash of the raw data used to generate these splits
