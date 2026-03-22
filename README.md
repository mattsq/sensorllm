# SensorLLM

Structured experiments on **image adapter approaches** for fusing aircraft sensor data with large language models.

## Overview

SensorLLM treats time-series aircraft sensor readings as images and uses visual adapter
architectures to bridge sensor modalities into an LLM's token space — enabling natural
language reasoning about sensor states, anomalies, and fault conditions.

```
Aircraft Sensor Data (IMU, vibration, pressure, ...)
         │
  [Signal → Image Transform]   (spectrogram, wavelet scalogram, recurrence plot)
         │
  Sensor Image (H × W)
         │
  [Sensor Encoder]             (ViT, ResNet)
         │
  Sensor Embeddings (N × D)
         │
  [Image Adapter]              (Linear Projection, Q-Former, Perceiver Resampler)
         │
  Token Embeddings (T × D_llm)
         │
  [LLM Backbone]               (LLaMA-3, Mistral, etc.)
         │
  Natural Language Output
```

The core research question: **which adapter architecture best transfers sensor information
into LLM reasoning for aircraft health monitoring tasks?**

## Adapter Approaches Compared

| Adapter | Style | Reference |
|---------|-------|-----------|
| Linear Projection | MLP bridge | LLaVA-1 |
| Q-Former | Cross-attention with learnable queries | BLIP-2 |
| Perceiver Resampler | Latent array cross-attention | Flamingo |
| MLP-Mixer | Token mixing without attention | — |

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo-url> && cd sensorllm
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure paths
cp .env.example .env
# Edit .env: set SENSORLLM_DATA_ROOT, SENSORLLM_OUTPUT_ROOT, WANDB_PROJECT

# 3. Run a baseline experiment
python scripts/train.py --config configs/experiments/exp001_linear_proj_vit.yaml

# 4. Evaluate
python scripts/evaluate.py \
    --config configs/experiments/exp001_linear_proj_vit.yaml \
    --checkpoint outputs/runs/exp001_linear_proj_vit/<timestamp>/best_model/
```

## Repository Structure

```
sensorllm/          Main Python package
  data/             Sensor readers, signal→image transforms, PyTorch datasets
  models/           Encoders, adapter architectures, LLM wrappers
  training/         Training loops, losses, callbacks
  evaluation/       Metrics, evaluators, benchmarks
  utils/            Config loading, logging, reproducibility
configs/            Versioned YAML experiment configs (base / adapters / experiments)
scripts/            CLI entry points (train, evaluate, infer, preprocess)
notebooks/          Exploratory analysis and result visualization
tests/              Pytest unit and integration tests
data/               Sensor data (raw/, processed/, spectrograms/) — NOT committed
outputs/            Experiment runs and checkpoints — NOT committed
docs/               Architecture docs, data format specs, experiment log
```

## Running Tests

```bash
pytest tests/                         # all tests
pytest tests/unit/ -m "not slow"      # fast unit tests only
pytest tests/ --cov=sensorllm         # with coverage
```

## Adding a New Adapter

1. Implement `SensorAdapter` base class in `sensorllm/models/adapters/my_adapter.py`
2. Register in `sensorllm/models/adapters/__init__.py`
3. Add config template in `configs/adapters/my_adapter.yaml`
4. Create experiment config in `configs/experiments/expNNN_my_adapter_*.yaml`
5. Add unit tests in `tests/unit/test_adapters.py`

See `CLAUDE.md` and `sensorllm/models/CLAUDE.md` for full conventions.

## Citation

```bibtex
@misc{sensorllm2025,
  title  = {SensorLLM: Image Adapter Approaches for Aircraft Sensor-LLM Fusion},
  year   = {2025},
}
```
