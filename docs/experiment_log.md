# Experiment Log

Chronological log of experiments. Add an entry for every new experiment config created.
Record the motivation, key differences from the prior experiment, and results when available.

---

## exp001 — Linear Projection + ViT-B/16 (Stage 1 Baseline)

**Config**: `configs/experiments/exp001_linear_proj_vit.yaml`
**Date**: —
**Status**: Not yet run

**Motivation**: Establish baseline performance with the simplest possible adapter
(two-layer MLP + adaptive average pool) and a standard ViT-B/16 encoder pretrained
on ImageNet. Mel spectrogram transform. Stage 1 alignment only.

**Key settings**: 10k steps, lr=1e-4, batch=32, 32 output tokens

**Results**: TBD

---

## exp002 — Q-Former + ViT-B/16 (Stage 1)

**Config**: `configs/experiments/exp002_qformer_vit.yaml`
**Date**: —
**Status**: Not yet run

**Motivation**: Test whether learnable cross-attention queries (Q-Former) improve over
the linear projection baseline. Same encoder and data config for fair comparison.

**Key differences from exp001**: Adapter = Q-Former (6-layer, 8-head, 32 queries)

**Results**: TBD

---

## exp003 — Perceiver Resampler + ResNet-50 (Stage 1)

**Config**: `configs/experiments/exp003_perceiver_resnet.yaml`
**Date**: —
**Status**: Not yet run

**Motivation**: Test Perceiver Resampler with a CNN spatial encoder (ResNet-50).
ResNet features may capture different sensor image structure than ViT patch tokens.

**Key differences from exp002**: Adapter = Perceiver (4-layer, 64 latents),
Encoder = ResNet-50 (layer4 features)

**Results**: TBD

---

*Add new entries here in the format above when creating new experiment configs.*
