# Adapter Architecture Taxonomy

Comparison of the four adapter approaches implemented in SensorLLM.

## Overview

All adapters sit between the time-series encoder and the LLM. They receive encoder
latent embeddings `(B, N, D_enc)` and produce fixed-length LLM token embeddings
`(B, T, D_llm)`. The fixed output length T is the key interface contract.

Adapters are agnostic to the encoder used — any encoder that outputs `(B, N, D_enc)`
can be paired with any adapter.

## 1. Linear Projection (`linear_projection`)

**Architecture**: Adaptive average pool (N → T tokens) → 2-layer MLP

**Strengths**:
- Simplest and fastest to train
- Fewest parameters
- Clear baseline

**Weaknesses**:
- Average pooling discards positional/temporal relationships
- Limited capacity for complex feature transformation
- No selective attention to task-relevant features

**When to use**: As a baseline in exp001; when training compute is constrained.

---

## 2. Q-Former (`qformer`)

**Reference**: BLIP-2 (Li et al., 2023) — adapted for time-series latents

**Architecture**: T learnable query tokens + multi-layer Transformer decoder
(cross-attention to encoder latents, self-attention between queries)

**Strengths**:
- Queries selectively extract task-relevant features from encoder output
- More expressive than linear projection
- Effective when encoder produces more patches than needed

**Weaknesses**:
- More parameters than linear projection
- Requires careful learning rate tuning for query tokens

**When to use**: exp002 with Transformer encoder; expected to improve over baseline
on complex anomaly patterns.

---

## 3. Perceiver Resampler (`perceiver`)

**Reference**: Flamingo (Alayrac et al., 2022) — adapted for time-series latents

**Architecture**: Fixed latent array → iterative cross-attention to encoder context
+ latent self-attention + FFN

**Strengths**:
- Strong compression for long latent sequences (many patches)
- Latent self-attention builds richer representations than Q-Former
- Handles variable-length encoder output gracefully

**Weaknesses**:
- More memory/compute per step than Q-Former
- More hyperparameters (latent_dim, ff_mult, n_layers)

**When to use**: exp003 with PatchTST encoder where N is large (overlapping patches
produce ~126 tokens for a 4096-sample window).

---

## 4. MLP-Mixer (`mlp_mixer`)

**Architecture**: Alternating token-mixing and channel-mixing MLPs → adaptive pool → projection

**Strengths**:
- No attention mechanism — computationally efficient
- Good for long input sequences
- Fully parallelizable

**Weaknesses**:
- Less expressive than attention-based adapters on complex tasks
- Token mixing MLPs tied to input sequence length

**When to use**: Efficiency baseline; useful when inference latency is critical.

---

## Comparison Table

| Adapter | Params | Key Mechanism | Best Paired With |
|---------|--------|---------------|-----------------|
| Linear Projection | Low | Avg pool + MLP | CNN1D (short N) |
| Q-Former | Medium | Learnable queries + cross-attn | Transformer (medium N) |
| Perceiver Resampler | Medium-High | Latent cross-attn + self-attn | PatchTST (long N) |
| MLP-Mixer | Medium | Token/channel mixing | Any (efficiency focus) |

T = n_output_tokens, N = n_patches from encoder, D = hidden dimension
