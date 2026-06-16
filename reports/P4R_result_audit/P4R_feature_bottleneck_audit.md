# P4R Feature Bottleneck Audit

## Current TMDC-v1 MOSI

| Modality | Encoder | Dim | T | Format |
|----------|---------|:--:|:--:|--------|
| Text | DeBERTa-large | 1024 | **1** | Pooled single vector |
| Audio | wav2vec2-large | 1024 | **1** | Pooled single vector |
| Vision | CLIP-ViT-B/32+proj | 1024 | **1** | Pooled single vector |

## Impact on sLSTM

- sLSTM with T=1: no temporal state transition → equivalent to FC+activation
- The "LSTM" in sLSTM is effectively unused
- Cannot demonstrate long-range dependency modeling
- Cannot model modality asynchrony or alignment

## MMSA Baseline Features (for comparison)

| Dataset | Text | Audio | Video | T |
|---------|------|-------|-------|:--:|
| MOSI (MMSA) | GloVe 300d / BERT 768d | COVAREP 74d | FACET 35d | ~50 |
| MOSI (MLCL) | BERT 768d | COVAREP 74d | FACET 35d | ~50 |

These are ALIGNED word-level/frame-level features, not pooled single vectors.

## Priority

**Route B (sequence features) is the critical path to 85%+.**

Without sequence features, no amount of model tuning will reach MMSA baseline levels.
The sLSTM was designed for temporal modeling — it needs sequences to prove its value.
