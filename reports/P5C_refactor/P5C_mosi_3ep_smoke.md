# P5C MOSI 3-Epoch Smoke Test Report

> Date: 2026-06-17
> Model: DeepText-xLSTM-AWAF Residual (ablation=none)
> Seed: 42, Batch: 32, LR: 1e-4
> Features: v3_T40 (text=1024, audio=768, vision=768)

## Training Progress

| Epoch | Train Loss | Val MAE | Val ACC2_NZ |
|-------|-----------|---------|-------------|
| 1 | 1.7563 | 1.3665 | 57.41% |
| 2 | 1.6743 | 1.2166 | 66.67% |
| 3 | 1.4890 | 1.0977 | 74.54% |

## Test Results (best epoch=3)

| Metric | Value |
|--------|-------|
| **ACC2_Non0 (reg_sign)** | **75.00%** |
| F1_Non0 (reg_sign) | 70.61% |
| ACC2_Has0 | 73.91% |
| MAE | 1.1227 |
| Corr | 0.5226 |
| ACC7 | 32.07% |
| ACC2_Non0 (cls) | 74.24% |
| F1_Non0 (cls) | 71.01% |

## AWAF Weights

| Modality | Mean | Std |
|----------|------|-----|
| Text (w_t) | 0.2056 | 0.0316 |
| Audio (w_a) | 0.1780 | 0.0473 |
| Vision (w_v) | 0.6164 | 0.0666 |

sum(w) max_dev = 1.19e-07 ✅

## Analysis

### Positive Signs
1. **Strong convergence**: val ACC2 increases +8.9% per epoch (57→67→75)
2. **Loss decreasing steadily**: 1.76 → 1.67 → 1.49
3. **No NaN or instability**
4. **AWAF weights valid**: sum(w)=1, non-collapsed

### Concerns (early stage)
1. **Vision weight high (0.62)**: Vision dominating the residual branch
2. **ACC2 at 75% at epoch 3**: Below P4W 78.8%, but trending up strongly
3. **MAE still high (1.12)**: Expected for epoch 3

### Judgment
- **3-epoch smoke is promising** — all systems functional
- **Rate of improvement suggests** 80%+ achievable by epoch 15-20
- **Need full 60-epoch training** for final assessment
- **Vision-heavy AWAF weights** may rebalance with more training

## Next Step
Full MOSI seed42 60-epoch training (already launched)
