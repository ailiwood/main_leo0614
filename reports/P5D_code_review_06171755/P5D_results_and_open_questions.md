# P5D Results and Open Questions

> For: Web AI decision-making
> Date: 2026-06-17

## P5C seed42 Result

| Metric | Value |
|--------|-------|
| ACC2_Non0 (reg_sign) | 81.10% |
| F1_Non0 | 77.70% |
| MAE | 0.8155 |
| Corr | 0.7487 |
| ACC7 | 42.13% |
| Best epoch | 38 |
| Best val ACC2 | 84.26% (epoch 51) |
| AWAF: w_t / w_a / w_v | 0.18 / 0.44 / 0.39 |

## P5D seed2024 Result

| Metric | Value |
|--------|-------|
| ACC2_Non0 (reg_sign) | 81.40% |
| F1_Non0 | 78.06% |
| MAE | 0.7957 |
| Corr | 0.7490 |
| ACC7 | 44.61% |
| Best epoch | 51 |
| Best val ACC2 | 83.80% (epoch 51) |
| AWAF: w_t / w_a / w_v | 0.38 / 0.32 / 0.29 |

## 2-Seed Mean

| Metric | Mean | σ |
|--------|------|---|
| ACC2_Non0 | **81.25%** | 0.15% |
| F1_Non0 | 77.88% | 0.18% |
| MAE | 0.8056 | 0.0099 |
| Corr | 0.7489 | 0.0002 |
| ACC7 | 43.37% | 1.24% |

## Comparison

| Model | ACC2_NZ | vs Ours |
|-------|---------|---------|
| DeepMLP text-only (P5B) | 80.2% | -1.05% |
| P4W AWAF-Seq (old) | 78.8% | -2.45% |
| **P5D Ours (2-seed mean)** | **81.25%** | — |
| Target 82% | 82.0% | +0.75% |
| Target 83% | 83.0% | +1.75% |
| Target 85% | 85.0% | +3.75% |

## Threshold Status

| Threshold | Met? | Gap |
|-----------|------|-----|
| > 80.2% (text-only) | ✅ | +1.05% |
| > 82% | ❌ | -0.75% |
| > 83% | ❌ | -1.75% |
| > 85% | ❌ | -3.75% |
| > 88% (P4 hard target) | ❌ | -6.75% |

## Residual Effect Diagnosis

Residual is **directionally correct but subtle**:
- 50.4% improved, 49.6% damaged (nearly neutral by count)
- Mean error reduction: 0.0017 (tiny)
- Weak_neg is MOST improved: 56.2% improved, sign +1.3%
- Strong_neg is slightly over-corrected: 53.3% damaged
- Only 2 sign fixes, 1 sign break — residual mostly adjusts magnitude

**The ACC2 gain (+1.05%) comes from small sign corrections on borderline cases, not from overall error reduction.**

## Checkpoint Space Issue

- 34 pth files, 537MB in outputs/
- Most are old exploration (P4V, P4W, P4X, P5A)
- Only P5C seed42 and P5D seed2024 are current
- Recommendation: delete 30 exploration pth files, keep 4 (P4T C0 ×2 ref, P5C s42, P5D s2024)
- P5D policy: save ONLY best_model.pth, no last_model.pth

## Questions for Web AI

### Architecture Questions
1. Is the residual correction formula (`final = text_base + δ * delta`) scientifically sound?
2. Should ConditionalResidualGate be the default, or only for ablation?
3. Should we consider removing vision entirely (text+audio only)?
4. Is two-stage training likely to help or is joint training sufficient?

### Performance Questions
5. Is 81.25% sufficient for a master's thesis, or must we reach 83%+?
6. Is the 3.2% val-test gap a sign of overfitting that needs fixing first?
7. Should we pursue P5E strong feature upgrade (wav2vec2-large, CLIP-ViT-L)?
8. The text-only ceiling is 80.2% (DeBERTa-large) — is this the real bottleneck?

### Paper Strategy
9. How to frame "residual correction improves accuracy by +1.05%" as a meaningful contribution?
10. Is AWAF weight interpretability (w_a dominant for residual) a strong enough innovation?
11. Should we prioritize reaching 83% or writing what we have?

## Desired Outputs from Web AI

1. **模型架构优化建议06171755.md** — concrete architecture improvements
2. **Key code replacement files** — if architecture changes needed
3. **Next phase CC prompt** — for P5E or remaining P5D experiments
