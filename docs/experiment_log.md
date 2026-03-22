# Experiment Log

Chronological log of experiments. Add an entry for every new experiment config created.
Record the motivation, key differences from the prior experiment, and results when available.

---

## exp001 — CNN1D Encoder + Linear Projection Adapter (Stage 1 Baseline)

**Config**: `configs/experiments/exp001_cnn1d_linear.yaml`
**Date**: —
**Status**: Not yet run

**Motivation**: Establish baseline performance with the simplest end-to-end pipeline.
Dilated 1D residual CNN encodes raw vibration sensor windows directly into temporal
patch embeddings (16 patches from a 4096-sample window). Linear projection (avg pool
+ MLP) maps to 32 LLM tokens. No image transform at any stage.

**Key settings**: 10k steps, lr=1e-4, batch=32, 32 output tokens, 16 encoder patches

**Results**: TBD

---

## exp002 — Transformer Encoder + Q-Former Adapter (Stage 1)

**Config**: `configs/experiments/exp002_transformer_qformer.yaml`
**Date**: —
**Status**: Not yet run

**Motivation**: Test whether a Transformer-based temporal encoder (patch size=64,
producing 64 patches) paired with Q-Former's selective cross-attention queries
improves over the CNN1D + linear projection baseline.

**Key differences from exp001**: Encoder = Transformer (64 patches), Adapter = Q-Former
(6-layer, 8-head, 32 learnable queries)

**Results**: TBD

---

## exp003 — PatchTST Encoder + Perceiver Resampler Adapter (Stage 1)

**Config**: `configs/experiments/exp003_patchtst_perceiver.yaml`
**Date**: —
**Status**: Not yet run

**Motivation**: Test the strongest combination: PatchTST's channel-independent
overlapping patching (patch_len=64, stride=32 → ~126 patches) paired with Perceiver
Resampler's expressive latent compression to 64 LLM tokens.

**Key differences from exp002**: Encoder = PatchTST (overlapping patches, ~126 patches),
Adapter = Perceiver Resampler (4-layer, 64 latents)

**Results**: TBD

---

*Add new entries here in the format above when creating new experiment configs.*
