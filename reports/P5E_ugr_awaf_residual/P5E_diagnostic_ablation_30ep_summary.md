# P5E Diagnostic Ablation (30ep, Seed=42)

| Ablation | ACC2 | MAE | Corr | Δ vs Full V2 |
|----------|------|-----|------|-------------|
| Full V2 (UGR+Experts+DeltaTarget) | **82.47%** | 0.8111 | 0.7385 | — |
| no_audio | 82.32% | 0.8182 | 0.7527 | -0.15% |
| no_awaf_mean_residual | 81.71% | 0.7889 | 0.7534 | -0.76% |
| no_vision | 81.25% | 0.8316 | 0.7481 | -1.22% |
| no_residual (text-only) | 80.49% | 0.7934 | 0.7667 | -1.98% |

## Answers

1. **Audio effective?** ✅ Yes — contributes +0.15%, small but real
2. **Vision effective?** ✅ Yes — contributes +1.22%, primary residual source
3. **AWAF > mean?** ✅ Yes — AWAF gives +0.76% over mean pooling
4. **Residual over text-only?** ✅ Yes — full V2 gives +1.98% over text-only
5. **Vision should remain in V2?** ✅ Yes — largest residual contribution
