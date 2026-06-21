# P6AB TAV Ablation Statistics
**Date**: 2026-06-21

## Paired Comparisons (F0 = Full TAV control)

### F0 vs F1: No Vision (T+A)
| Metric | F0 | F1 | Difference | 95% CI |
|--------|----|----|----|----|
| ACC2_Non0 | 78.32% | 76.65% | +1.67pp | [+0.61, +2.73] pp |
| MAE | 0.7345 | 0.7421 | -0.0077 | [-0.0138, -0.0016] |
| Pred Corr | ！ | ！ | 0.9012 | ！ |

- McNemar: chi2=0.00, A-only=231, B-only=233
- Interpretation: F0 significantly BETTER than F1 (CI excludes 0).

### F0 vs F2: No Audio (T+V)
| Metric | F0 | F2 | Difference | 95% CI |
|--------|----|----|----|----|
| ACC2_Non0 | 78.32% | 78.72% | -0.39pp | [-1.37, +0.58] pp |
| MAE | 0.7345 | 0.7155 | +0.0189 | [+0.0125, +0.0253] |
| Pred Corr | ！ | ！ | 0.9535 | ！ |

- McNemar: chi2=7.56, A-only=198, B-only=146
- Interpretation: No significant difference (CI includes 0).

### F0 vs F3: Global Static Weights
| Metric | F0 | F3 | Difference | 95% CI |
|--------|----|----|----|----|
| ACC2_Non0 | 78.32% | 79.30% | -0.97pp | [-1.79, -0.15] pp |
| MAE | 0.7345 | 0.7064 | +0.0281 | [+0.0223, +0.0338] |
| Pred Corr | ！ | ！ | 0.9543 | ！ |

- McNemar: chi2=0.57, A-only=120, B-only=133
- Interpretation: F0 significantly WORSE than F3 (CI excludes 0).

### F0 vs F4: Fixed Mean (1/3 each)
| Metric | F0 | F4 | Difference | 95% CI |
|--------|----|----|----|----|
| ACC2_Non0 | 78.32% | 79.30% | -0.97pp | [-1.79, -0.15] pp |
| MAE | 0.7345 | 0.7064 | +0.0281 | [+0.0223, +0.0338] |
| Pred Corr | ！ | ！ | 0.9543 | ！ |

- McNemar: chi2=0.57, A-only=120, B-only=133
- Interpretation: F0 significantly WORSE than F4 (CI excludes 0).

### F0 vs F5: AWAF No Interaction
| Metric | F0 | F5 | Difference | 95% CI |
|--------|----|----|----|----|
| ACC2_Non0 | 78.32% | 75.96% | +2.37pp | [+1.49, +3.25] pp |
| MAE | 0.7345 | 0.7489 | -0.0144 | [-0.0187, -0.0102] |
| Pred Corr | ！ | ！ | 0.9528 | ！ |

- McNemar: chi2=3.07, A-only=162, B-only=131
- Interpretation: F0 significantly BETTER than F5 (CI excludes 0).

### F0 vs E1: LSTM Temporal Encoder
| Metric | F0 | E1 | Difference | 95% CI |
|--------|----|----|----|----|
| ACC2_Non0 | 78.32% | 77.78% | +0.55pp | [-0.49, +1.58] pp |
| MAE | 0.7345 | 0.7096 | +0.0248 | [+0.0180, +0.0317] |
| Pred Corr | ！ | ！ | 0.9561 | ！ |

- McNemar: chi2=36.99, A-only=282, B-only=154
- Interpretation: No significant difference (CI includes 0).

## F0 AWAF Weight Analysis

| Stat | w_t (text) | w_a (audio) | w_v (vision) |
|------|-----------|------------|-------------|
| Mean | 0.4345 | 0.2633 | 0.3021 |
| Std | 0.0307 | 0.0323 | 0.0192 |
| Min | 0.3193 | 0.2112 | 0.2351 |
| Max | 0.4843 | 0.3759 | 0.3597 |
| Avg Entropy | 1.0715 | ！ | ！ |
