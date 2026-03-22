# SensorLLM

Structured experiments on **adapter approaches** for fusing aircraft sensor data directly
with large language models — encoding sensor time-series in the temporal domain without
any image intermediary.

## Overview

SensorLLM encodes raw windowed sensor signals through a time-series encoder (CNN1D,
Transformer, PatchTST) to produce temporal patch embeddings, which an adapter
architecture compresses into a fixed-length sequence of LLM token embeddings.

```
Raw Sensor Signal (B, C, L)     — windowed time-series
        │
  [Time-Series Encoder]         CNN1D | Transformer | PatchTST
        │
  Latent Embeddings (B, N, D)   — N temporal patches
        │
  [Sensor Adapter]              Linear Projection | Q-Former | Perceiver | MLP-Mixer
        │
  Token Embeddings (B, T, D_llm)
        │
  [LLM Backbone]                LLaMA-3, Mistral, etc.
        │
  Natural Language Output
```

The core research question: **which encoder + adapter combination best transfers aircraft
sensor information into LLM reasoning for anomaly detection and fault diagnosis?**

## Experiments

| Experiment | Encoder | Adapter | Notes |
|-----------|---------|---------|-------|
| exp001 | CNN1D (dilated residual) | Linear Projection | Fastest baseline |
| exp002 | Transformer (patch-based) | Q-Former | Attention-based encoder + selective adapter |
| exp003 | PatchTST (channel-independent) | Perceiver Resampler | Strongest combination |

## Quick Start

```bash
# 1. Set up environment
git clone <repo-url> && cd sensorllm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure paths
cp .env.example .env
# Edit .env: set SENSORLLM_DATA_ROOT, SENSORLLM_OUTPUT_ROOT, WANDB_PROJECT

# 3. Run a baseline experiment
python scripts/train.py --config configs/experiments/exp001_cnn1d_linear.yaml

# 4. Evaluate
python scripts/evaluate.py \
    --config configs/experiments/exp001_cnn1d_linear.yaml \
    --checkpoint outputs/runs/exp001_cnn1d_linear/<timestamp>/best_model/
```

## Repository Structure

```
sensorllm/          Main Python package
  data/             Sensor readers, windowing, PyTorch datasets (raw signal pipeline)
  models/           Time-series encoders, adapter architectures, LLM wrappers
  training/         Training loops, losses, callbacks
  evaluation/       Metrics, evaluators, benchmarks
  utils/            Config loading, logging, reproducibility
configs/            Versioned YAML experiment configs
scripts/            CLI entry points (train, evaluate, infer, preprocess)
notebooks/          Exploratory analysis and result visualization
tests/              Pytest unit and integration tests
data/               Sensor data (NOT committed)
outputs/            Experiment runs and checkpoints (NOT committed)
docs/               Architecture docs, data format specs, experiment log
```

## Running Tests

```bash
pytest tests/                         # all tests
pytest tests/unit/ -m "not slow"      # fast unit tests only
pytest tests/unit/test_encoders.py    # encoder tests
pytest tests/unit/test_adapters.py    # adapter tests
pytest tests/ --cov=sensorllm         # with coverage
```

## Adding a New Encoder

1. Implement `SensorEncoder` base class: input `(B, C, L)` → output `(B, N, D)`
2. Register in `sensorllm/models/encoders/__init__.py`
3. Add config template in `configs/base/model_base.yaml` or experiment config
4. Add unit tests in `tests/unit/test_encoders.py`

See `CLAUDE.md` and `sensorllm/models/CLAUDE.md` for full conventions.

## Citation

```bibtex
@misc{sensorllm2025,
  title  = {SensorLLM: Adapter Approaches for Aircraft Sensor-LLM Fusion},
  year   = {2025},
}
```
