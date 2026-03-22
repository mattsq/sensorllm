# Adapter Architecture Taxonomy

Comparison of the four adapter approaches implemented in SensorLLM.

## Overview

All adapters implement the same interface: take sensor encoder embeddings `(B, N, D_enc)`
and produce fixed-length LLM token embeddings `(B, T, D_llm)`.

## 1. Linear Projection (`linear_projection`)

**Reference**: LLaVA-1 (Liu et al., 2023)

**Architecture**: Adaptive average pool (N → T tokens) → 2-layer MLP

**Strengths**:
- Simplest and fastest to train
- Fewest parameters
- Good baseline

**Weaknesses**:
- Average pooling discards spatial relationships
- Limited capacity for feature transformation
- No selective attention to task-relevant sensor features

**When to use**: As a baseline; when training compute is limited.

---

## 2. Q-Former (`qformer`)

**Reference**: BLIP-2 (Li et al., 2023)

**Architecture**: T learnable query tokens + multi-layer Transformer decoder
(cross-attention to sensor embeddings, self-attention between queries)

**Strengths**:
- Queries can selectively extract task-relevant features
- More expressive than linear projection
- Well-studied in vision-language literature

**Weaknesses**:
- More parameters than linear projection
- Slower to converge (requires more alignment pretraining)
- Query tokens must be initialized carefully

**When to use**: Main comparison target; expected to improve over linear projection on
complex anomaly patterns where selective attention matters.

---

## 3. Perceiver Resampler (`perceiver`)

**Reference**: Flamingo (Alayrac et al., 2022)

**Architecture**: Fixed latent array → iterative cross-attention to sensor context
+ latent self-attention + FFN

**Strengths**:
- Strong compression for long sensor sequences (handles arbitrary N)
- Latent self-attention builds richer representations than Q-Former
- Scales well to high-frequency sensors with many patches

**Weaknesses**:
- More memory/compute per step than Q-Former
- More hyperparameters (latent_dim, ff_mult, n_layers)

**When to use**: High-frequency vibration data with long windows; when Q-Former
plateaus and richer latent representations are needed.

---

## 4. MLP-Mixer (`mlp_mixer`)

**Reference**: MLP-Mixer (Tolstikhin et al., 2021) adapted for sensor-LLM bridging

**Architecture**: Alternating token-mixing and channel-mixing MLPs → adaptive pool → projection

**Strengths**:
- No attention mechanism — computationally efficient
- Good for long input sequences
- Parallelizable (no sequential dependencies)

**Weaknesses**:
- Token mixing MLPs are input-length-specific (may need fixed N from encoder)
- Less expressive than attention-based adapters on complex tasks

**When to use**: When inference latency is critical; as an efficiency baseline vs.
attention-based adapters.

---

## Comparison Table

| Adapter | Params | Complexity | Key Mechanism | Reference Style |
|---------|--------|-----------|---------------|-----------------|
| Linear Projection | Low | O(T·D) | Avg pool + MLP | LLaVA-1 |
| Q-Former | Medium | O(T·N·D) | Learnable queries + cross-attn | BLIP-2 |
| Perceiver Resampler | Medium-High | O(T·N·D) | Latent cross-attn + self-attn | Flamingo |
| MLP-Mixer | Medium | O(T·N) | Token/channel mixing MLPs | MLP-Mixer |

T = n_output_tokens, N = n_patches from encoder, D = hidden dimension
