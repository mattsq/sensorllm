# Training — Agent Guide

## Purpose

This subpackage implements the training loop. It wraps HuggingFace `Trainer` (or a
custom loop) with sensor-LLM-specific callbacks, loss handling, and the two-stage
freezing protocol.

## Entry Point

Training is launched via `scripts/train.py`:

```bash
python scripts/train.py --config configs/experiments/exp001_linear_proj_vit.yaml

# Override specific config values (dot-notation):
python scripts/train.py \
    --config configs/experiments/exp001_linear_proj_vit.yaml \
    --override training.learning_rate=1e-4 training.max_steps=20000
```

The script: loads config → sets seed → builds model + datasets → instantiates
`SensorLLMTrainer` → calls `trainer.train()`.

## Two-Stage Protocol

**Stage 1** (`training.stage: 1`) — adapter alignment:
- Encoder frozen ✓, LLM frozen ✓, adapter trainable ✓
- Typical: 10k–50k steps on a sensor description pretraining dataset
- Goal: align sensor features to LLM token distribution

**Stage 2** (`training.stage: 2`) — instruction fine-tuning:
- Encoder frozen ✓, LLM trained with LoRA ✓, adapter trainable ✓
- Initialize adapter from Stage 1 checkpoint: set `model.adapter.pretrained_adapter`
- Goal: instruction-following on sensor Q&A tasks

`SensorLLMTrainer._apply_stage_freezing()` reads `config.training.stage` and calls
`.requires_grad_(False)` on the appropriate modules.

## Callbacks

`MetricsLoggerCallback` — logs per-step metrics to W&B and `metrics.jsonl` in run dir

`BestModelCallback` — saves `checkpoint-{step}/` and updates `best_model/` symlink
when validation loss improves

`FrozenParamCallback` — verifies frozen parameters stay frozen (catches bugs where
a frozen module accidentally receives gradients)

## Output Directory Layout

Each training run creates:
```
outputs/runs/{experiment_name}/{YYYY-MM-DD_HHMM}/
├── config.yaml          # Frozen copy of the experiment config used
├── metrics.jsonl        # One JSON line per logging step
├── logs/                # Text logs
├── checkpoint-1000/     # Checkpoints saved every save_steps
├── checkpoint-2000/
└── best_model -> checkpoint-2000/   # Symlink to best checkpoint
```

## Key Config Fields (training block)

```yaml
training:
  stage: 1                          # 1 = align, 2 = instruct finetune
  max_steps: 10000
  learning_rate: 1e-4
  warmup_steps: 500
  gradient_accumulation_steps: 4
  bf16: true                        # Use bfloat16 mixed precision
  save_steps: 1000
  eval_steps: 500
  logging_steps: 50
  output_dir: $SENSORLLM_OUTPUT_ROOT
```
