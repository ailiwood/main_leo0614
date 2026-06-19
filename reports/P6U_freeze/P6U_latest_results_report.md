# P6U: Latest Freeze-Ready Results Report

**Date**: 2026-06-20  
**Branch**: `p6u-strict-freeze-cleanup`  
**Status**: ✅ 14 Models freeze-ready

---

## A. MOSI Results (5 Models)

| # | Model | Modality | ACC2_Non0 | F1_Non0 | MAE | Corr | Best Ep | Source |
|---|-------|----------|-----------|---------|-----|------|---------|--------|
| M1 | **Main (ours)** | text_audio | **88.72%** | 86.64% | 0.635 | 0.851 | ? | P6K |
| B1 | TFN-lite | text_audio | 77.59% | 71.01% | 1.078 | 0.614 | ? | P6Q |
| B2 | MulT-lite | text_audio | 77.13% | 71.80% | 0.942 | 0.641 | ? | P6P |
| B3 | SelfMM-lite | text_audio | 76.68% | 73.11% | 1.024 | 0.614 | ? | P6P |
| B4 | LMF-lite | text_audio | 75.61% | 72.41% | 1.045 | 0.586 | ? | P6Q |

## B. MOSEI Results (9 Models)

| # | Model | Modality | ACC2_Non0 | F1_Non0 | MAE | Corr | ACC7 | Best Ep | Source |
|---|-------|----------|-----------|---------|-----|------|------|---------|--------|
| M2 | **Main (ours)** | text_audio | **87.86%** | 90.25% | 0.519 | 0.800 | 54.75% | 12 | P6T |
| D1 | Main (ours) diag | text_only | 88.22% | 90.57% | 0.509 | 0.813 | 55.60% | 5 | P6S_repair |
| B5 | MISA-lite | text_audio | 82.67% | 86.62% | 0.621 | 0.687 | 49.94% | 12 | P6S_repair5 |
| B6 | SelfMM-lite | text_audio | 82.60% | 86.33% | 0.623 | 0.682 | 49.16% | 12 | P6S_repair5 |
| B7 | MulT-lite | text_audio | 82.39% | 86.17% | 0.611 | 0.705 | 48.90% | 11 | P6S_repair5 |
| B8 | LMF-lite | text_audio | 81.85% | 85.67% | 0.624 | 0.682 | 48.97% | 8 | P6S_repair5 |
| B9 | TFN-lite | text_audio | 81.48% | 85.69% | 0.638 | 0.664 | 48.71% | 7 | P6S_repair5 |
| B10 | MLCL-lite | text_audio | 80.66% | 84.24% | 0.656 | 0.649 | 47.29% | 6 | P6S_repair5 |
| B11 | DLF-lite | text_audio | 77.90% | 82.98% | 0.733 | 0.555 | 44.18% | 2 | P6S_repair5 |

## C. Excluded Models

| Model | Dataset | Reason |
|-------|---------|--------|
| MMIM-lite | MOSEI | Implementation byte-identical to MISA-lite |
| TAV (text_av) | MOSEI | Vision features all-zero |
| P6S_repair4 8 baselines | MOSEI | Systematic COVAREP -inf collapse |
| P6N/P6O 8 baselines | MOSI | Training collapse (constant predictions) |

## D. Multimodal Gain Summary

| Dataset | text_only | text_audio | Delta |
|---------|-----------|------------|-------|
| MOSI | 83.99% | **88.72%** | **+4.73%** ✅ |
| MOSEI | 88.22% | 87.86% | -0.36% ≈ 0 |

MOSEI COVAREP 74d audio adds no discriminative gain over RoBERTa-large text.  
Thesis: "Text-dominant signal; cross-modal robustness without degradation."

## E. Path Verification

All 14 models have:
- ✅ result.json
- ✅ best_model.pth
- ✅ predictions_test.csv (M2 generated via re-eval)

## F. Credibility

All 14 models: Credibility **A**, can_enter_paper **True**.  
All results from real training + unified `utils/metrics.py`.  
No smoke/debug/subset results included.

Generated: 2026-06-20
