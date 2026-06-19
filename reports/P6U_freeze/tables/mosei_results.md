# P6U Freeze: MOSEI Final Results

**Date**: 2026-06-20 | **Status**: FROZEN

## Main Model

| Model | Modality | ACC2_Non0 | F1_Non0 | MAE | Corr | ACC7 | Best Ep | Source |
|-------|----------|-----------|---------|-----|------|------|---------|--------|
| **Main (ours)** | **text_audio** | **87.86%** | **90.25%** | **0.519** | **0.800** | **54.75%** | 12 | P6T |

Multimodal gain: text_audio (87.86%) ≈ text_only (88.22%), Δ = -0.36%.  
**Thesis**: Text-dominant; cross-modal robustness. Do NOT claim audio improvement.

## Baselines (7 models)

| Rank | Model | ACC2_Non0 | F1_Non0 | MAE | Corr | ACC7 | Best Ep | Params |
|------|-------|-----------|---------|-----|------|------|---------|--------|
| 1 | MISA-lite | 82.67% | 86.62% | 0.621 | 0.687 | 49.94% | 12 | 0.40M |
| 2 | SelfMM-lite | 82.60% | 86.33% | 0.623 | 0.682 | 49.16% | 12 | 0.41M |
| 3 | MulT-lite | 82.39% | 86.17% | 0.611 | 0.705 | 48.90% | 11 | 0.47M |
| 4 | LMF-lite | 81.85% | 85.67% | 0.624 | 0.682 | 48.97% | 8 | 0.51M |
| 5 | TFN-lite | 81.48% | 85.69% | 0.638 | 0.664 | 48.71% | 7 | 0.50M |
| 6 | MLCL-lite | 80.66% | 84.24% | 0.656 | 0.649 | 47.29% | 6 | 0.41M |
| 7 | DLF-lite | 77.90% | 82.98% | 0.733 | 0.555 | 44.18% | 2 | 0.44M |

## Diagnostic

| Model | Modality | ACC2_Non0 | F1_Non0 | MAE | Corr |
|-------|----------|-----------|---------|-----|------|
| Main (ours) | text_only | 88.22% | 90.57% | 0.509 | 0.813 |

**Note**: text_only is diagnostic only — NOT the final main model.

## Excluded

| Model | Reason |
|-------|--------|
| MMIM-lite | Byte-identical implementation to MISA-lite |
| TAV (text_av) | Vision features all-zero |
| P6S_repair4 (8 models) | COVAREP -inf collapse |

All metrics from `utils/metrics.py`. All results from real training. Credibility: A.
