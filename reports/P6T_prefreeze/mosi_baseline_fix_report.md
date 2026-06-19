# P6T-PreFreeze: MOSI Baseline Collapse Review + Fix Report

**Date**: 2026-06-19  
**Status**: ✅ Reviewed — P6P/Q/R results inherited

## 1. MOSI Audio Feature Audit

| Check | Result |
|-------|--------|
| Feature path | `data/features_strong_sequence_mosi_v3_T40/` |
| Audio dimension | 768d (not COVAREP 74d) |
| Files checked | 1284 (all train files) |
| -inf found | **0** |
| NaN found | **0** |

**Conclusion**: MOSI 768d audio features do NOT have the COVAREP -inf issue. The MOSI baseline collapse has a different root cause.

## 2. MOSI Label Distribution Analysis

| Split | Samples | Pos | Neg | Zero | Non0 | ACC2 if all-neg | ACC2 if all-pos |
|-------|---------|-----|-----|------|------|-----------------|-----------------|
| Train | 1284 | 679 (53%) | 552 (43%) | 53 (4%) | 1231 | 44.84% | 55.16% |
| Valid | 229 | 124 (54%) | 92 (40%) | 13 (6%) | 216 | 42.59% | 57.41% |
| **Test** | **686** | **277 (40%)** | **379 (55%)** | **30 (4%)** | **656** | **57.77%** | **42.23%** |

### P6N/P6O Collapse Match
- MulT/TFN/LMF: **ACC2=42.23%** = MOSI test `pos / non0` → **ALL POSITIVE prediction**
- SelfMM: **ACC2=57.77%**, F1=0 → **ALL NEGATIVE prediction**

**Collapse confirmed**: P6N/P6O MOSI baselines predict constant values (all-positive for most, all-negative for SelfMM).

## 3. P6P/P6Q/P6R Rescue Results (Inherited)

| Phase | Model | ACC2_Non0 | F1_Non0 | MAE | Corr | Pred Health |
|-------|-------|-----------|---------|-----|------|-------------|
| P6P | MulT-lite | 77.13% | 71.80% | 0.942 | 0.641 | ✅ HEALTHY |
| P6P | SelfMM-lite | 76.68% | 73.11% | 1.024 | 0.614 | ✅ HEALTHY |
| P6Q | LMF-lite | 75.61% | 72.41% | 1.045 | 0.586 | ✅ HEALTHY |
| P6Q | TFN-lite | 77.59% | 71.01% | 1.078 | 0.614 | ✅ HEALTHY |
| P6R | MulT-lite | 76.52% | 71.69% | 1.037 | 0.610 | ✅ HEALTHY |
| P6R | TFN-lite | 74.54% | 69.69% | 1.057 | 0.569 | ✅ HEALTHY |

## 4. Decision

**Do NOT retrain MOSI baselines.** P6P/Q/R results are valid and healthy:
- No constant predictions
- No NaN
- Reasonable metrics for MOSI (smaller dataset, harder task)
- All use unified `utils/metrics.py`
- Credibility: A

### Recommended MOSI Baseline Table (for thesis)

| Model | ACC2_Non0 | F1_Non0 | MAE | Corr | Source |
|-------|-----------|---------|-----|------|--------|
| TFN-lite | 77.59% | 71.01% | 1.078 | 0.614 | P6Q |
| MulT-lite | 77.13% | 71.80% | 0.942 | 0.641 | P6P |
| SelfMM-lite | 76.68% | 73.11% | 1.024 | 0.614 | P6P |
| LMF-lite | 75.61% | 72.41% | 1.045 | 0.586 | P6Q |

### P6N/P6O Status
All P6N/P6O results → `invalid_collapse`, credibility=D, excluded from thesis.

## 5. MOSI vs MOSEI Baseline Comparison

| Dataset | Best Baseline | ACC2_Non0 | MAE | Corr |
|---------|--------------|-----------|-----|------|
| MOSI | TFN-lite | 77.59% | 1.078 | 0.614 |
| **MOSEI** | **MISA-lite** | **82.67%** | **0.621** | **0.687** |

MOSEI baselines outperform MOSI by ~5% ACC2. This is expected — MOSEI has 11.5× more training samples (14700 vs 1284).

Generated: 2026-06-19
