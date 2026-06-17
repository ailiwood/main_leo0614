# P5D Stage Decision

> Date: 2026-06-17
> Phase: P5D — Residual Stability & Performance Sprint

## Completed

| Task | Status | Key Finding |
|------|--------|-------------|
| seed=2024 training | ✅ | ACC2=81.40%, 2-seed mean=81.25% |
| Stability verified | ✅ | σ=0.15% — architecture is stable |
| Checkpoint audit | ✅ | 34 pth files, 537MB, cleanup plan ready |
| Residual analysis | ✅ | Weak_neg most improved (+1.3% sign) |
| ConditionalResidualGate | ✅ | Implemented, tested (9/9), 75K params |
| Sample reweight | ✅ | Implemented in losses.py |
| Two-stage trainer | ✅ | Script ready, not yet run |
| MOSEI status | ✅ | Blocked, dimensions compatible |

## Not Yet Run

| Task | Priority | Est. Time |
|------|----------|-----------|
| Diagnostic ablation (4×30ep) | High | ~45 min |
| ConditionalResidualGate seed42 | High | ~13 min |
| Two-stage training seed42 | Medium | ~30 min |
| Weak_neg reweight seed42 | Medium | ~13 min |
| ConditionalGate seed2024 (if gate works) | Low | ~13 min |

## Current Best

| Model | ACC2_NZ | MAE | Corr |
|-------|---------|-----|------|
| P5C one-stage (2-seed mean) | **81.25%** | 0.8056 | 0.7489 |
| DeepMLP text-only | 80.2% | 0.885 | 0.733 |
| P4W AWAF-Seq | 78.8% | 0.994 | 0.645 |

## Decision

1. **Architecture validated**: Residual fusion stably beats text-only (+1.05%)
2. **Not at 82% yet**: Need to run remaining P5D experiments
3. **Do NOT freeze model**: At least run ConditionalResidualGate before decision
4. **Do NOT enter P5E** (strong feature upgrade): Run P5D sprint first
5. **Do NOT start MOSEI**: Still blocked

## Recommended Next Actions

1. Execute diagnostic ablation (no_audio, no_vision, no_awaf_mean, no_residual) × 30ep
2. Train ConditionalResidualGate seed42 60ep
3. If gate improves → train gate seed2024
4. If weak_neg still bottleneck → try reweight
5. If text base degradation suspected → try two-stage
6. If best candidate > 82.5% → add seed=1234
