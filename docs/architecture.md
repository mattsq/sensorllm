# SensorLLM Architecture

## Overview

SensorLLM is a multimodal model that bridges aircraft sensor time-series data with
large language models using image adapter approaches.

```
Aircraft Sensor Data (time-series)
         │
         ▼
┌────────────────────┐
│  Signal → Image    │  spectrogram / wavelet scalogram / recurrence plot
│  Transform         │
└────────────────────┘
         │  image (H × W)
         ▼
┌────────────────────┐
│  Sensor Encoder    │  ViT-B/16, ResNet-50, CNN1D
│                    │  → patch/spatial embeddings (N × D_enc)
└────────────────────┘
         │  (B, N, D_enc)
         ▼
┌────────────────────┐
│  Sensor Adapter    │  Linear Projection | Q-Former | Perceiver | MLP-Mixer
│                    │  → fixed-length token embeddings (T × D_llm)
└────────────────────┘
         │  (B, T, D_llm)   T = n_output_tokens (constant)
         ▼
┌────────────────────┐
│  Text Prompt       │  tokenized → embedding lookup → (B, S, D_llm)
│  Embeddings        │
└────────────────────┘
         │  concat: [sensor tokens | prompt tokens]  (B, T+S, D_llm)
         ▼
┌────────────────────┐
│  LLM Backbone      │  LLaMA-3, Mistral, etc.
│  (Causal LM)       │  → logits over vocabulary
└────────────────────┘
         │
         ▼
  Generated Text / Loss
```

## Key Design Decisions

### Fixed-Length Adapter Output

All adapters output exactly `n_output_tokens` token embeddings regardless of input
sensor sequence length. This is the key contract that makes adapters drop-in swappable.

### Two-Stage Training

Training follows the LLaVA / BLIP-2 two-stage protocol:

1. **Stage 1 — Adapter Alignment**: Only the adapter is trained. Encoder and LLM are
   frozen. The task is predicting sensor descriptions. This aligns the sensor feature
   space with the LLM's expected token distribution.

2. **Stage 2 — Instruction Fine-tuning**: Encoder stays frozen. The adapter and LLM
   (via LoRA) are trained together on instruction-following tasks (Q&A, anomaly
   description, fault diagnosis).

### Sensor Token Injection

The LLM tokenizer is extended with a special `<sensor>` placeholder token. During
forward pass, wherever `<sensor>` appears in the input_ids, it is replaced with the
`n_output_tokens` sensor token embeddings from the adapter. This allows flexible
positioning of sensor tokens within multi-turn prompts.

## Component Interfaces

See `sensorllm/models/CLAUDE.md` for detailed interface contracts.
