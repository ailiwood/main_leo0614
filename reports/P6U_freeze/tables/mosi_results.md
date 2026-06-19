# P6U Freeze: MOSI Final Results

**Date**: 2026-06-20 | **Status**: FROZEN

## Main Model

| Model | Modality | ACC2_Non0 | F1_Non0 | MAE | Corr | Source |
|-------|----------|-----------|---------|-----|------|--------|
| **Main (ours)** | **text_audio** | **88.72%** | **86.64%** | **0.635** | **0.851** | P6K |

Multimodal gain: text_audio (88.72%) > text_only (83.99%), Δ = +4.73%. ✅ Audio helps on MOSI.

## Baselines (4 models)

| Rank | Model | ACC2_Non0 | F1_Non0 | MAE | Corr | Source |
|------|-------|-----------|---------|-----|------|--------|
| 1 | TFN-lite | 77.59% | 71.01% | 1.078 | 0.614 | P6Q |
| 2 | MulT-lite | 77.13% | 71.80% | 0.942 | 0.641 | P6P |
| 3 | SelfMM-lite | 76.68% | 73.11% | 1.024 | 0.614 | P6P |
| 4 | LMF-lite | 75.61% | 72.41% | 1.045 | 0.586 | P6Q |

## Excluded

| Model | Reason |
|-------|--------|
| P6N/P6O (8 models) | Training collapse — constant predictions |

All metrics from `utils/metrics.py`. Credibility: A.
