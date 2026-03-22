# SensorLLM — Agent Guide

## Project Overview

SensorLLM experiments with **image adapter approaches** for fusing aircraft sensor data
with large language models. The pipeline: raw sensor time-series → image transform
(spectrogram / wavelet / recurrence plot) → sensor encoder → adapter → LLM.

**Research goal**: determine which adapter architecture (linear projection, Q-Former,
Perceiver Resampler, MLP-Mixer) best transfers visual representations of aircraft sensor
data into LLM reasoning about anomalies, fault diagnosis, and operational state narration.

## Repository Layout

```
sensorllm/      Main Python package (models, data, training, evaluation, utils)
configs/        Versioned YAML experiment configs (base/, adapters/, experiments/)
scripts/        CLI entry points: train.py, evaluate.py, infer.py, preprocess_sensors.py
notebooks/      Jupyter notebooks for EDA and result visualization
tests/          Pytest test suite (unit/ and integration/)
data/           Sensor data files — NOT committed (gitignored)
outputs/        Experiment run outputs and checkpoints — NOT committed (gitignored)
docs/           Architecture docs, data format specs, experiment narrative log
```

## Environment Setup

```bash
# Create and activate virtualenv
python -m venv .venv && source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Configure environment variables
cp .env.example .env
# Edit .env and set: SENSORLLM_DATA_ROOT, SENSORLLM_OUTPUT_ROOT, WANDB_PROJECT, HF_TOKEN
```

Key environment variables:
- `SENSORLLM_DATA_ROOT` — absolute path to the `data/` directory
- `SENSORLLM_OUTPUT_ROOT` — absolute path to the `outputs/` directory
- `WANDB_PROJECT` — Weights & Biases project name for experiment tracking
- `HF_TOKEN` — HuggingFace token (required for gated models like LLaMA)

## Running Experiments

```bash
# Train with a specific experiment config
python scripts/train.py --config configs/experiments/exp001_linear_proj_vit.yaml

# Override individual config values (dot-notation)
python scripts/train.py --config configs/experiments/exp001_linear_proj_vit.yaml \
    --override training.learning_rate=5e-5 training.max_steps=20000

# Evaluate a checkpoint
python scripts/evaluate.py \
    --config configs/experiments/exp001_linear_proj_vit.yaml \
    --checkpoint outputs/runs/exp001_linear_proj_vit/2024-01-15_1400/best_model/

# Single-sample inference
python scripts/infer.py \
    --checkpoint outputs/runs/.../best_model/ \
    --sensor-file data/raw/flight_001.h5 \
    --prompt "Describe any anomalies in the vibration sensor data."

# Batch preprocess raw sensor files → spectrograms
python scripts/preprocess_sensors.py \
    --input-dir data/raw/ \
    --output-dir data/spectrograms/ \
    --transform spectrogram
```

## Running Tests

```bash
pytest tests/                              # all tests
pytest tests/unit/ -m "not slow"           # fast unit tests only
pytest tests/unit/test_adapters.py -v      # specific test file
pytest tests/ --cov=sensorllm --cov-report=html   # with coverage report
```

## Adding a New Adapter Architecture

1. Create `sensorllm/models/adapters/my_adapter.py` inheriting `SensorAdapter`
2. Implement `forward(sensor_embeddings, attention_mask=None) -> Tensor` and `n_output_tokens`
3. Register in `sensorllm/models/adapters/__init__.py` ADAPTER_REGISTRY
4. Add `configs/adapters/my_adapter.yaml` with default hyperparameters
5. Create `configs/experiments/expNNN_my_adapter_<encoder>.yaml` experiment config
6. Add unit tests in `tests/unit/test_adapters.py`

See `sensorllm/models/CLAUDE.md` for the full adapter interface specification.

## Adding a New Sensor Transform

1. Create `sensorllm/data/transforms/my_transform.py` inheriting `BaseTransform`
2. Implement `__call__(signal: np.ndarray) -> np.ndarray` (output: float32 array in [0,1])
3. Register in `sensorllm/data/transforms/__init__.py` TRANSFORM_REGISTRY
4. Visualize in `notebooks/02_transform_comparison.ipynb`

See `sensorllm/data/CLAUDE.md` for transform conventions.

## Config System

All hyperparameters live in versioned YAML files under `configs/`. Configs use `_base_`
inheritance keys. Loading: `sensorllm.utils.config.load_config(path)` → dataclass.

See `configs/CLAUDE.md` for the full schema and naming conventions.

## Experiment Tracking & Output Layout

- All runs log to Weights & Biases (`WANDB_PROJECT` in `.env`)
- Each run lands in: `outputs/runs/{experiment_name}/{YYYY-MM-DD_HHMM}/`
- Run directory contains: `config.yaml` (frozen), `metrics.jsonl`, `logs/`, checkpoints
- `best_model/` is symlinked to the best checkpoint by validation loss

## Code Conventions

- Python 3.10+, type annotations on all public functions and class methods
- `black` formatting (line length 100), `ruff` linting, `mypy` type checking
- Docstrings: Google style for all public classes and functions
- All randomness via `sensorllm.utils.reproducibility.set_seed(seed)`
- Logging via `sensorllm.utils.logging.get_logger(__name__)` — never bare `print()`
- No hardcoded paths: use config values or `os.environ` with clear error messages

## Key Design Patterns

**Config-driven**: every hyperparameter lives in a YAML config file, not in source code.

**Adapter interface**: all adapters implement `SensorAdapter.forward(sensor_embeddings)`
returning token embeddings of fixed length `n_output_tokens` — making them drop-in
swappable without touching the LLM input construction logic.

**Transform interface**: all sensor→image transforms implement
`__call__(signal: np.ndarray) -> np.ndarray` returning a float32 array of shape `(H, W)`
or `(C, H, W)` with values in [0, 1].

**Dataset contract**: all datasets return dicts with keys `sensor_image`, `input_ids`,
`attention_mask`, `labels` — the trainer sees a uniform interface regardless of sensor
type or transform used.

**Two-stage training**:
- Stage 1 — adapter alignment: freeze encoder + LLM, train adapter only
- Stage 2 — instruction fine-tuning: freeze encoder, train adapter + LLM (via LoRA)
