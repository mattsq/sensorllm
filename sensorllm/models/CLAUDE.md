# Models — Agent Guide

## Purpose

This subpackage defines all neural network components. The architecture is a
three-stage pipeline: **Time-Series Encoder** → **Adapter** → **LLM**.
No image conversion is used at any point.

## Architecture Overview

```
Sensor Signal (B × C × L)      — windowed raw time-series, no image transform
        │
  [SensorEncoder]               CNN1D | Transformer | PatchTST
        │
  Latent Embeddings             shape: (B, N, D_enc)
        │                       N = number of temporal patches
        │
  [SensorAdapter]               Linear Projection | Q-Former | Perceiver | MLP-Mixer
        │
  Token Embeddings              shape: (B, T, D_llm)
        │                       T = n_output_tokens (FIXED — adapter output is constant length)
  [LLM Backbone]                LLaMA-3, Mistral, etc.
        │
  Generated text / Loss
```

## Encoder Interface Contract

Every encoder MUST inherit `SensorEncoder` and implement:

```python
def forward(self, signals: torch.Tensor) -> torch.Tensor:
    # signals: (B, C, L)  →  returns: (B, N_patches, output_dim)
    ...

@property
def output_dim(self) -> int:
    ...
```

- Input `(B, C, L)`: B=batch, C=sensor channels, L=window length in samples
- Output `(B, N, D)`: N temporal patch embeddings of dimension D=output_dim

## Encoder Registry

| Key | Class | Description |
|-----|-------|-------------|
| `cnn1d` | `CNN1DSensorEncoder` | Dilated 1D residual CNN; fastest, good baseline |
| `transformer` | `TransformerSensorEncoder` | Patch-based 1D Transformer; captures long-range temporal dependencies |
| `patchtst` | `PatchTSTSensorEncoder` | Channel-independent patching + Transformer (Nie et al. 2023) |

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
    # Fixed — never varies at runtime
    ...
```

The fixed-length output is the key contract that makes adapters drop-in swappable.

## Adapter Registry

| Key | Class | Style | Notes |
|-----|-------|-------|-------|
| `linear_projection` | `LinearProjectionAdapter` | Avg pool + MLP | Fastest baseline |
| `qformer` | `QFormerAdapter` | Learnable queries + cross-attn | BLIP-2 style |
| `perceiver` | `PerceiverResamplerAdapter` | Latent cross-attn + self-attn | Flamingo style |
| `mlp_mixer` | `MLPMixerAdapter` | Token/channel mixing | Attention-free, efficient |

## Two-Stage Training Protocol

**Stage 1 — Alignment** (config: `training.stage: 1`):
- Encoder trainable ✓, adapter trainable ✓, LLM frozen ✓
- Task: predict sensor signal descriptions from raw time-series
- Goal: train encoder to produce informative latents; train adapter to map them into LLM token space

**Stage 2 — Instruction Fine-tuning** (config: `training.stage: 2`):
- Encoder frozen ✓, adapter trainable ✓, LLM trained with LoRA ✓
- Initialize from Stage 1 checkpoint
- Task: sensor Q&A, anomaly description, fault diagnosis

## Adding a New Encoder

1. Create `encoders/my_encoder.py` inheriting `SensorEncoder`
2. Implement `forward(signals: Tensor) -> Tensor` — input `(B, C, L)`, output `(B, N, D)`
3. Expose `output_dim: int` property
4. Register in `encoders/__init__.py` ENCODER_REGISTRY
5. Add unit test in `tests/unit/test_encoders.py` using a random `(B, C, L)` tensor

## SensorLLM Top-Level Model (`sensorllm_model.py`)

Wires the three components and handles:
- Sensor token injection into the LLM input sequence via `<sensor>` placeholder
- Generation API: `model.generate(sensor_signals, prompt_ids, prompt_mask)`
- `_encode_sensor(sensor_signals)`: convenience method for encoder → adapter pass
