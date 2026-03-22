# SensorLLM — Agent Guide

## Project Overview

SensorLLM experiments with **adapter approaches** for fusing aircraft sensor data
directly with large language models. The pipeline: raw sensor time-series → time-series
encoder → adapter → LLM. **No image conversion is used** — sensor data is encoded
directly in the temporal domain.

**Research goal**: determine which combination of time-series encoder (CNN1D, Transformer,
PatchTST) and adapter architecture (linear projection, Q-Former, Perceiver Resampler,
MLP-Mixer) best transfers sensor information into LLM reasoning about anomalies, fault
diagnosis, and operational state narration.

## Pipeline

```
Raw Sensor Signal (B, C, L)     — windowed time-series
        │
  [Time-Series Encoder]         CNN1D | Transformer | PatchTST
        │
  Latent Embeddings (B, N, D)   — N temporal patches, D = encoder hidden dim
        │
  [Sensor Adapter]              Linear Projection | Q-Former | Perceiver | MLP-Mixer
        │
  Token Embeddings (B, T, D_llm) — T = n_output_tokens (fixed)
        │
  [LLM Backbone]                LLaMA-3, Mistral, etc.
```

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
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env: set SENSORLLM_DATA_ROOT, SENSORLLM_OUTPUT_ROOT, WANDB_PROJECT, HF_TOKEN
```

Key environment variables:
- `SENSORLLM_DATA_ROOT` — absolute path to the `data/` directory
- `SENSORLLM_OUTPUT_ROOT` — absolute path to the `outputs/` directory
- `WANDB_PROJECT` — Weights & Biases project name
- `HF_TOKEN` — HuggingFace token (required for gated models like LLaMA)

## Running Experiments

```bash
# Train
python scripts/train.py --config configs/experiments/exp001_cnn1d_linear.yaml

# Override config values (dot-notation)
python scripts/train.py --config configs/experiments/exp001_cnn1d_linear.yaml \
    --override training.learning_rate=5e-5 training.max_steps=20000

# Evaluate
python scripts/evaluate.py \
    --config configs/experiments/exp001_cnn1d_linear.yaml \
    --checkpoint outputs/runs/exp001_cnn1d_linear/2024-01-15_1400/best_model/

# Single-sample inference
python scripts/infer.py \
    --checkpoint outputs/runs/.../best_model/ \
    --sensor-file data/raw/flight_001.h5 \
    --prompt "Describe any anomalies in the vibration sensor data."
```

## Running Tests

```bash
pytest tests/                              # all tests
pytest tests/unit/ -m "not slow"           # fast unit tests only
pytest tests/unit/test_encoders.py -v      # encoder tests
pytest tests/ --cov=sensorllm --cov-report=html
```

## Adding a New Time-Series Encoder

1. Create `sensorllm/models/encoders/my_encoder.py` inheriting `SensorEncoder`
2. Implement `forward(signals: Tensor) -> Tensor` — input `(B, C, L)`, output `(B, N, D)`
3. Expose `output_dim: int` property
4. Register in `sensorllm/models/encoders/__init__.py` ENCODER_REGISTRY
5. Add unit tests in `tests/unit/test_encoders.py`

See `sensorllm/models/CLAUDE.md` for the full encoder interface specification.

## Adding a New Adapter Architecture

1. Create `sensorllm/models/adapters/my_adapter.py` inheriting `SensorAdapter`
2. Implement `forward(sensor_embeddings, attention_mask=None) -> Tensor` and `n_output_tokens`
3. Register in `sensorllm/models/adapters/__init__.py` ADAPTER_REGISTRY
4. Add `configs/adapters/my_adapter.yaml`
5. Create an experiment config in `configs/experiments/expNNN_*.yaml`
6. Add unit tests in `tests/unit/test_adapters.py`

See `sensorllm/models/CLAUDE.md` for the full adapter interface specification.

## Config System

All hyperparameters live in versioned YAML files under `configs/`. Configs use `_base_`
inheritance. Loading: `sensorllm.utils.config.load_config(path)` → dict.

See `configs/CLAUDE.md` for the full schema and naming conventions.

## Experiment Tracking & Output Layout

- All runs log to Weights & Biases (`WANDB_PROJECT` in `.env`)
- Each run lands in: `outputs/runs/{experiment_name}/{YYYY-MM-DD_HHMM}/`
- Run directory: `config.yaml` (frozen), `metrics.jsonl`, `logs/`, checkpoints
- `best_model/` symlinked to best checkpoint by validation loss

## Code Conventions

- Python 3.10+, type annotations on all public functions
- `black` (line length 100), `ruff` linting, `mypy` type checking
- Google-style docstrings on all public classes/functions
- All randomness via `sensorllm.utils.reproducibility.set_seed(seed)`
- Logging via `sensorllm.utils.logging.get_logger(__name__)` — never `print()`
- No hardcoded paths: use config values or environment variables

## Key Design Patterns

**No image intermediary**: sensor data is always `(B, C, L)` — raw windowed time-series.
Never convert to spectrograms or other 2D representations.

**Adapter interface**: all adapters implement `SensorAdapter.forward(sensor_embeddings)`
returning fixed-length token embeddings (`n_output_tokens`).

**Encoder interface**: all encoders implement `SensorEncoder.forward(signals)` accepting
`(B, C, L)` and returning `(B, N, D)` temporal patch embeddings.

**Dataset contract**: all datasets return `{"sensor_signal": (C, L), "input_ids": ...,
"attention_mask": ..., "labels": ...}`.

**Two-stage training**:
- Stage 1 — alignment: freeze LLM, train encoder + adapter
- Stage 2 — instruction fine-tuning: freeze encoder, train adapter + LLM (LoRA)
