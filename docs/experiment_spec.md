# Experiment Spec: Bias Metrics Under Embedding Swaps + Full Telemetry

## Goal
Quantify how bias/fairness measurements change when you keep the same prompts and comparisons but change the embedding representation (layer, pooling, model type). Capture high-resolution telemetry to explain variance, regressions, and performance costs.

## Core Questions

1. Embedding fragility
- Sensitivity to layer choice
- Sensitivity to pooling (CLS / last / mean / attention-weighted)
- Sensitivity to embedding model (base LM vs sentence encoder)

2. Cross-metric agreement
- Correlation between embedding-based bias and probability-based bias
- Optional: correlation with generation-based bias

3. Where bias shows up
- Which layers/heads contribute most
- Whether bias is concentrated or distributed

## Experimental Design

Fixed counterfactual prompt pairs are evaluated through multiple metric families, while sweeping a grid of embedding extraction variants.

### Inputs

Sentence pairs or prompts in stereotyped vs anti-stereotyped form, plus group swaps.
Example: “X is a doctor” vs “X is a nurse” under demographic name/group substitution.

The same sample set is reused for all metrics to allow direct correlation.

### Embedding Variants (Independent Variable)

For each model, extract sentence representations using a matrix like:

- Layer: 0 … L (early/mid/late)
- Source: hidden state / attention output / MLP output (optional)
- Pooling: CLS, last token, mean tokens, attention-weighted mean
- Normalization: none vs L2
- Embedding model: base LM embeddings vs external sentence encoders

This yields a controlled “representation swap” while everything else stays constant.

### Metrics (Dependent Variables)

1. Embedding-based bias metrics
- WEAT/SEAT-style association tests on contextual sentence embeddings
- Directional projection scores (e.g., gender direction components)

2. Probability-based bias metrics
- Pseudo-log-likelihood differences for stereo vs anti-stereo sentence pairs
- Per-token logprob contribution breakdown

3. Generation-based bias metrics (optional in MVP)
- Differences in attribute words, toxicity/regard, or stereotypical associations in completions under group swaps

## Telemetry (Differentiator)

Every run logs:

- System + reproducibility: git hash, environment, seeds, model hashes
- Performance: tokenize/forward/metric timing, throughput, batch sizes
- Resource: peak VRAM, CPU RAM, GPU utilization, clocks, power draw, temperature
- Model internals (optional “full mode”): activation norms per layer, attention entropy per head, layer ablation deltas

This lets you answer whether bias score shifts are real or an artifact of representation choice or compute noise.

## Outputs / Deliverables

- Layer x pooling heatmaps of bias scores per model
- Stability plots: variance/CI across embedding variants
- Correlation matrix: embedding-metrics vs PLL vs generation metrics
- Attribution report: layers/heads most responsible for bias deltas
- A reusable evaluation harness with pluggable embedding extraction

## MVP

- 1 model + 1 dataset subset
- 3 embedding variants (early/mid/late layer, mean pooling)
- 2 metrics (SEAT-like + PLL)
- Telemetry: timing + VRAM + GPU power/clock

## Extensions

- Add sentence encoders (E5/BGE/MiniLM) as alternative embeddings
- Add per-layer ablations + head masking
- Add multi-lingual prompts (Greek/English)
