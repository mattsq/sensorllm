# SensorLLM Architecture

## Overview

SensorLLM is a multimodal model that bridges aircraft sensor time-series data directly
with large language models. Sensor data is encoded in the **temporal domain** — no
image conversion step is used.

```
Aircraft Sensor Data (time-series)
         │  raw windowed signal (B, C, L)
         ▼
┌────────────────────┐
│  Time-Series       │  CNN1D, Transformer, PatchTST
│  Encoder           │  → temporal patch embeddings (B, N, D_enc)
└────────────────────┘
         │  (B, N, D_enc)
         ▼
┌────────────────────┐
│  Sensor Adapter    │  Linear Projection | Q-Former | Perceiver | MLP-Mixer
│                    │  → fixed-length token embeddings (B, T, D_llm)
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
│  LLM Backbone      │  LLaMA-3, Mistral, etc. (causal LM)
│                    │  → logits over vocabulary
└────────────────────┘
         │
         ▼
  Generated Text / Loss
```

## Key Design Decisions

### Direct Time-Series Encoding

Sensor data is **never converted to images**. Time-series encoders (CNN1D, Transformer,
PatchTST) process raw windowed signals `(B, C, L)` directly into latent embeddings
`(B, N, D)`. This preserves temporal resolution, avoids information loss from image
transforms, and removes a hyperparameter-heavy preprocessing step.

### Fixed-Length Adapter Output

All adapters output exactly `n_output_tokens` token embeddings regardless of input
patch count N. This makes adapters drop-in swappable without touching the LLM input
construction logic.

### Two-Stage Training

1. **Stage 1 — Alignment**: encoder + adapter train together with LLM frozen.
   Task: generate natural language descriptions from sensor windows.
   Teaches the encoder/adapter to produce LLM-compatible representations.

2. **Stage 2 — Instruction Fine-tuning**: encoder frozen, adapter + LLM (LoRA) trained.
   Task: sensor Q&A, anomaly description, fault diagnosis.

### Sensor Token Injection

The LLM tokenizer is extended with a `<sensor>` placeholder token. During forward pass,
sensor tokens from the adapter replace the `<sensor>` position in the input embedding
sequence, allowing flexible multi-turn prompt construction.

## Component Interfaces

See `sensorllm/models/CLAUDE.md` for detailed interface contracts.

## Encoder Comparison

| Encoder | Mechanism | Temporal Coverage | Speed |
|---------|-----------|------------------|-------|
| CNN1D | Dilated 1D residual convolutions | Local + multi-scale | Fast |
| Transformer | Self-attention over non-overlapping patches | Global | Medium |
| PatchTST | Channel-independent overlapping patches + Transformer | Global + overlap | Medium |
