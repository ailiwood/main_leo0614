# P6A Baseline Metric Audit

## Our Primary Metric: ACC2_Non0 (reg_sign)

- **Definition**: Binary accuracy excluding label=0. Negative = label<0, Positive = label>=0.
- **Prediction**: reg_sign (sign of regression output), not cls_logit
- **Excludes**: label exactly 0
- **Reported as**: percentage (0-100)

## Secondary Metrics
- ACC2_Has0: Same but including label=0 (reported separately)
- ACC2_Non0_cls: Using cls_logit instead of reg_sign (reported separately)
- F1_Non0, MAE, Corr, ACC7

## Critical Warnings

1. **Do NOT mix Non0 with Has0**: Many papers report "ACC-2" without specifying. We must assume Non0 for fair comparison unless proven otherwise.
2. **Do NOT mix negative/non-negative with negative/positive**: Our Non0 excludes zero. Has0 includes zero.
3. **CASP is TTA**: Must be in separate table row, labeled "CASP (TTA)".
4. **MLCL**: Need to verify from original paper (Zhuang et al. 2025, TMM). Paper-reported only until locally reproduced.

## Baseline Sources

| Model | Source | Reproduced? | Directly Comparable? |
|-------|--------|-------------|---------------------|
| TFN | Literature | No | ⚠️ Paper values only |
| LMF | Literature | No | ⚠️ Paper values only |
| MulT | Literature | No | ⚠️ Paper values only |
| MISA | Literature | No | ⚠️ Paper values only |
| Self-MM | Literature | No | ⚠️ Paper values only |
| MLCL | Literature (Zhuang 2025) | No | ⚠️ Need to verify |
| CASP | Literature (Guo 2025) | No | ❌ TTA, separate table |
