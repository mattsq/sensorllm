# Configs — Agent Guide

## Purpose

All experiment hyperparameters live here as versioned YAML files. No magic numbers
in source code. Every training run receives a config that is frozen and saved with
outputs for full reproducibility.

## Config Composition via `_base_`

Configs use a `_base_` list for inheritance. The loader deep-merges in listed order,
then applies the current file on top:

```yaml
_base_:
  - configs/base/data_base.yaml
  - configs/base/model_base.yaml
  - configs/base/training_base.yaml
  - configs/adapters/qformer.yaml
# Keys below override anything from the base configs
experiment_name: exp002_qformer_vit
model:
  encoder:
    name: vit_b16
```

Loading: `sensorllm.utils.config.load_config("configs/experiments/exp002_qformer_vit.yaml")`

## Full Config Schema

```
experiment_name     str       Unique experiment identifier (used for output dir naming)
seed                int       Global random seed

data:
  data_root         str       Path to data directory (or $SENSORLLM_DATA_ROOT)
  sensors           list[str] Sensor modalities: ["imu", "vibration", "pressure", ...]
  transform         str       Transform key from TRANSFORM_REGISTRY
  window_size       int       Samples per window
  hop_size          int       Hop between windows
  image_size        int       Target H=W for resize after transform
  batch_size        int
  num_workers       int

model:
  encoder:
    name            str       Key from ENCODER_REGISTRY ("vit_b16", "resnet50", "cnn1d")
    pretrained      str|null  HuggingFace model ID or local path
    freeze          bool      Whether to freeze during all training
  adapter:
    name            str       Key from ADAPTER_REGISTRY
    n_output_tokens int       Fixed output token count
    hidden_dim      int       Adapter internal width (adapter-specific meaning)
    n_layers        int       Number of layers (Q-Former, Perceiver, MLP-Mixer)
    n_heads         int       Number of attention heads (Q-Former, Perceiver)
  llm:
    name            str       HuggingFace model ID (e.g., "meta-llama/Llama-3-8B-Instruct")
    torch_dtype     str       "bfloat16" | "float16" | "float32"
    freeze          bool
    lora:
      enabled       bool
      r             int       LoRA rank
      alpha         float     LoRA scaling
      target_modules list[str]  Module names to apply LoRA to

training:
  stage             int       1 (alignment) or 2 (instruction fine-tuning)
  max_steps         int
  learning_rate     float
  warmup_steps      int
  gradient_accumulation_steps int
  bf16              bool
  fp16              bool
  save_steps        int
  eval_steps        int
  logging_steps     int
  output_dir        str       Path or $SENSORLLM_OUTPUT_ROOT

evaluation:
  metrics           list[str]  Metric keys (e.g., ["bleu", "rouge", "f1"])
  generation_max_new_tokens int
```

## Experiment Naming Convention

Format: `expNNN_<adapter>_<encoder>[_<variant>].yaml`

```
exp001_linear_proj_vit.yaml     # Stage 1 baseline
exp002_qformer_vit.yaml         # Stage 1 Q-Former
exp003_perceiver_resnet.yaml    # Stage 1 Perceiver + ResNet
exp004_qformer_vit_stage2.yaml  # Stage 2 follow-on from exp002
```

- **Increment NNN sequentially** — never reuse a number
- Create a new config for every distinct experiment (never modify after a run)
- Add a brief entry to `docs/experiment_log.md` when creating a new experiment config

## Directory Layout

```
base/               Default values inherited by all experiments
  data_base.yaml
  model_base.yaml
  training_base.yaml
adapters/           Per-adapter hyperparameter defaults
  linear_projection.yaml
  qformer.yaml
  perceiver.yaml
experiments/        Full experiment configs (one per experiment run)
  exp001_*.yaml
  exp002_*.yaml
  ...
```
