# P5E V2 2-Seed Summary

| Seed | ACC2_NZ | F1_NZ | MAE | Corr | ACC7 | Best Epoch |
|------|---------|-------|-----|------|------|------------|
| 42 | **82.47%** | 78.66% | 0.8111 | 0.7385 | 41.11% | 45 |
| 2024 | 81.86% | 78.48% | 0.8486 | 0.7316 | 40.23% | 23 |
| **Mean** | **82.17%** | **78.57%** | **0.8299** | **0.7351** | **40.67%** | — |

## Comparison

| Model | ACC2_NZ | MAE | Corr |
|-------|---------|-----|------|
| DeepMLP text-only | 80.2% | 0.885 | 0.733 |
| P4W AWAF-Seq | 78.8% | 0.994 | 0.645 |
| P5D V1 (2-seed) | 81.25% | 0.8056 | 0.7489 |
| **P5E V2 (2-seed)** | **82.17%** | **0.8299** | **0.7351** |

Δ vs P5D: ACC2 +0.92%, MAE +0.0243 (slightly worse), Corr -0.0138

## Judgment

- ✅ 82.17% > 82% threshold
- ❌ 82.17% < 83% — not yet strong candidate
- ❌ 82.17% < 82.5% — don't add 3rd seed
- MAE slightly degraded vs P5D — regression quality trade-off
- V2 is current best candidate but model NOT frozen
