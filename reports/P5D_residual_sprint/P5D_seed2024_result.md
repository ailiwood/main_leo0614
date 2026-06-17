# P5D Seed=2024 Result

> Date: 2026-06-17
> Model: DeepText-xLSTM-AWAF Residual (ablation=none)
> Seed: 2024, Batch: 32, LR: 1e-4
> Best epoch: 51, Best val ACC2: 83.80%

## Test Results

| Metric | Seed=42 | Seed=2024 | 2-Seed Mean |
|--------|---------|-----------|-------------|
| **ACC2_Non0 (reg)** | 81.10% | **81.40%** | **81.25%** |
| F1_Non0 | 77.70% | 78.06% | 77.88% |
| ACC2_Has0 | 79.45% | 79.45% | 79.45% |
| MAE | 0.8155 | **0.7957** | **0.8056** |
| Corr | 0.7487 | 0.7490 | 0.7489 |
| ACC7 | 42.13% | **44.61%** | **43.37%** |
| ACC2_Non0 (cls) | 80.95% | 81.10% | 81.03% |

## AWAF Weights

| Modality | Seed42 | Seed2024 |
|----------|--------|----------|
| Text (w_t) | 0.1753±0.0778 | 0.3820±0.3044 |
| Audio (w_a) | 0.4382±0.2781 | 0.3232±0.2488 |
| Vision (w_v) | 0.3864±0.3202 | 0.2947±0.1012 |

## Stability Assessment

**✅ Residual architecture is STABLE across seeds**
- Both seeds > DeepMLP text-only (80.2%)
- 2-seed mean = 81.25% with σ = 0.15% (very tight)
- MAE consistently better than text-only (0.8056 < 0.885)
- Corr consistently better (0.7489 > 0.733)

## Judgment Matrix

| Criterion | Value | Status |
|-----------|-------|--------|
| seed2024 ≥ 80.2% | 81.40% | ✅ Passed |
| 2-seed mean ≥ 81.0% | 81.25% | ✅ Enter sprint |
| 2-seed mean ≥ 82.5% | 81.25% | ❌ No 3rd seed |
| 2-seed mean < 80.5% | 81.25% | ❌ Not this case |
