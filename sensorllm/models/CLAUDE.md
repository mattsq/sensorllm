# Models — Agent Guide

## Purpose

This subpackage defines all neural network components. The architecture follows a
three-stage pipeline: **Encoder** → **Adapter** → **LLM**.

## Architecture Overview

```
SensorImage (B × C × H × W)
       │
  [SensorEncoder]           e.g., ViT-B/16, ResNet-50, CNN1D
       │
  sensor_embeddings         shape: (B, N_patches, encoder_dim)
       │
  [SensorAdapter]           e.g., LinearProjection, QFormer, PerceiverResampler
       │
  token_embeddings          shape: (B, n_tokens, llm_dim)
       │                    n_tokens is FIXED — adapter compresses to constant length
  [LLM Backbone]            e.g., LLaMA-3-8B, Mistral-7B
       │
  generated text
```

## Adapter Interface Contract

Every adapter MUST inherit `SensorAdapter` and implement:

```python
def forward(
    self,
    sensor_embeddings: torch.Tensor,   # (B, N_patches, encoder_dim)
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:                     # (B, n_output_tokens, llm_hidden_dim)
    ...

@property
def n_output_tokens(self) -> int:
    # Fixed number of tokens this adapter produces — never varies at runtime
    ...
```

The fixed-length output is the key contract that makes adapters drop-in swappable
without changing the LLM input construction logic in `SensorLLMModel`.

## Adapter Taxonomy

| Key | Class | File | Style | Notes |
|-----|-------|------|-------|-------|
| `linear_projection` | `LinearProjectionAdapter` | `linear_projection.py` | MLP + avg pool | Fast baseline, low capacity |
| `qformer` | `QFormerAdapter` | `qformer.py` | Learnable queries + cross-attn | BLIP-2 style; good selectivity |
| `perceiver` | `PerceiverResamplerAdapter` | `perceiver.py` | Latent array + cross-attn | Flamingo style; strong compression |
| `mlp_mixer` | `MLPMixerAdapter` | `mlp_mixer.py` | Token/channel mixing | No attention; efficient for long seqs |

## Encoder Interface Contract

Every encoder MUST inherit `SensorEncoder` and implement:

```python
def forward(self, images: torch.Tensor) -> torch.Tensor:
    # images: (B, C, H, W) → returns: (B, N_patches, output_dim)
    ...

@property
def output_dim(self) -> int:
    ...
```

## Two-Stage Training Protocol

Standard training follows LLaVA / BLIP-2 convention:

**Stage 1 — Adapter Alignment** (config: `training.stage: 1`):
- Freeze encoder and LLM; train adapter only
- Task: predict sensor descriptions from spectrogram images
- Goal: teach adapter to map sensor features into LLM token distribution

**Stage 2 — Instruction Fine-tuning** (config: `training.stage: 2`):
- Freeze encoder; train adapter + LLM with LoRA
- Initialize adapter from Stage 1 checkpoint (`model.adapter.pretrained_adapter`)
- Task: sensor Q&A, anomaly description, fault diagnosis

## Adding a New Encoder

1. Create `encoders/my_encoder.py` inheriting `SensorEncoder`
2. Implement `forward(images) -> (B, N, D)` and `output_dim` property
3. Register in `encoders/__init__.py` ENCODER_REGISTRY under a string key
4. Add unit test in `tests/unit/test_adapters.py` using a random tensor input

## Adding a New Adapter

See root `CLAUDE.md` for step-by-step instructions.
The key constraint: `n_output_tokens` must be a fixed integer regardless of input shape.

## SensorLLM Top-Level Model (`sensorllm_model.py`)

`SensorLLMModel` wires the three components and handles:
- Sensor token embedding injection into the LLM input sequence
- `<sensor>` placeholder token handling (sensor tokens replace placeholder positions)
- Generation API: `model.generate(sensor_images, prompt_ids, prompt_mask)`
- Checkpoint save/load splitting encoder/adapter/LLM weights separately
