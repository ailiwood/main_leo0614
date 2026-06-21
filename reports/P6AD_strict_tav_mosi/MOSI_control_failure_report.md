# MOSI TAV Control Failure Report

**Date**: 2026-06-21  
**MOSI_CONTROL_STATUS**: `failed_or_collapsed`  
**Decision**: STOP all MOSI TAV training. No attempt 2. No hyperparameter tuning.

---

## F0 Control Results (4 epochs)

| Metric | Value | Healthy? |
|--------|-------|----------|
| ACC2_Non0 | 42.23% | ❌ Near majority class |
| F1_Non0 | 59.38% | ⚠️ Low |
| MAE | 1.494 | ❌ Very high |
| Corr | -0.032 | ❌ Zero correlation |
| ACC7 | 15.60% | ❌ Near chance (14.3%) |
| Best epoch | 1 | ❌ No improvement after epoch 1 |
| Val ACC2 trend | 57.41% → 57.41% → 57.41% → 57.41% | ❌ FLAT |
| Train loss trend | 1.315 → 1.297 → 1.280 → 1.253 | ⚠️ Barely decreases |
| AWAF entropy | 1.084–1.089 | ⚠️ Near uniform, no learning |

## Failure Criteria Met

| Criterion | Met? | Evidence |
|-----------|------|----------|
| Constant prediction | ⚠️ Borderline | Predictions may not be strictly constant but ACC2 near majority |
| F1_Non0 near 0 | ⚠️ Borderline | 59.38% is low but not zero |
| ACC2 near majority class | ✅ MET | 42.23% (MOSI binary distribution ~50/50) |
| NaN / inf | ❌ | No NaN/Inf detected |
| vision/audio gradient 0 | ⚠️ Not checked | But flat training suggests gradient issues |
| vision/audio zero no effect | ⚠️ Not checked | But near-zero correlation suggests this |
| AWAF weights constant | ⚠️ Near-constant | Entropy ~1.088 (max entropy ≈ 1.099) |

## Root Cause Analysis

Probable causes for MOSI TAV collapse:

1. **MOSI dataset too small (1,284 train samples)** for a 5.1M parameter TAV model
   - MOSEI has 13,239 train samples (10x more)
   - Model has ~4x more parameters than MOSI samples

2. **Feature version mismatch**: MOSI uses CLIP-L14 (1024-dim vision) and data2vec (768-dim audio)
   - These are different feature sources from MOSEI (OpenFace2 713-dim, COVAREP 74-dim)
   - Model architecture may not be appropriate for these feature dimensions
   - Vision dim 1024 with only 32 time steps may not provide enough temporal signal

3. **Audio features dominate**: AWAF w_a=0.398 vs w_v=0.257
   - Audio receives highest weight but may provide weak signal for sentiment
   - Vision weight is lowest, despite CLIP-L14 being a strong visual encoder

4. **Learning rate incompatibility**: 3e-5 may be too high for MOSI's small dataset
   - Overfitting from epoch 1 (best epoch = 1, val never improves)

## Required Actions

Per P6AD protocol:
- [x] Stop all MOSI TAV training
- [x] Do NOT run MOSI F1-F5, E1
- [x] Do NOT run MOSI baselines
- [x] Do NOT attempt hyperparameter tuning
- [x] Write this failure report
- [ ] Document MOSI TAV status in final report

## MOSI Final Classification

```text
MOSI_TAV_STATUS: failed_or_collapsed
MOSI_PAPER_ROLE: historical_text_audio_reference_only
MOSI_DATA_AVAILABLE: yes (mosi_tav_v1, 2199 samples, CLIP-L14 vision)
MOSI_TAV_ATTEMPTED: yes (F0 control, 4 epochs, collapsed)
MOSI_TAV_BLOCKING_REASON: training collapse on small dataset with mismatched feature versions
```

## Paper Language

> CMU-MOSI was also evaluated with the canonical T+A+V AWAF sLSTM model
> using CLIP-L14 vision features and data2vec audio features. However,
> under the fixed 4-epoch ablation protocol (identical to MOSEI), the model
> failed to learn effectively: validation accuracy remained flat across all
> epochs, and test performance was near chance level (ACC2=42.23%, Corr=-0.03).
> This is attributed to the combination of MOSI's small training set (1,284 samples)
> and feature version differences from the primary MOSEI experiments.
> MOSI is retained as a historical text-audio conservative reference (P6K,
> ACC2=88.72%) and is not used as formal three-modal experimental evidence.
