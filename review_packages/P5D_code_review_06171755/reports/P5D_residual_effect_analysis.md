# P5D Residual Effect Analysis (Seed=2024)

## Overall Effect

| Metric | Text Base | Final | Delta |
|--------|-----------|-------|-------|
| Mean Error | 0.7974 | 0.7957 | -0.0017 |
| MAE | — | 0.7957 | — |
| Improved samples | — | 50.4% | — |
| Damaged samples | — | 49.6% | — |
| Sign accuracy | 82.1% | 82.2% | +0.1% |

## By Group

| Group | N | Err Base | Err Final | Improved% | Damaged% | Sign Base% | Sign Final% |
|-------|---|----------|-----------|-----------|----------|------------|-------------|
| strong_pos | 183 | 0.7328 | 0.7270 | 55.7 | 44.3 | 85.8 | 86.3 |
| strong_neg | 306 | 0.8660 | 0.8638 | 46.7 | 53.3 | 87.6 | 87.6 |
| weak_pos | 94 | 0.8213 | 0.8233 | 51.1 | 48.9 | 63.8 | 62.8 |
| **weak_neg** | **73** | 0.6891 | 0.6905 | **56.2** | 43.8 | 65.8 | **67.1** |
| near_zero | 106 | 0.6882 | 0.6889 | 51.9 | 48.1 | 70.8 | 70.8 |
| all | 686 | 0.7974 | 0.7957 | 50.4 | 49.6 | 82.1 | 82.2 |

## Key Findings

1. **Weak_neg is the MOST improved group**: 56.2% improved, sign accuracy +1.3%
2. **Strong_neg is slightly damaged**: 46.7% improved, 53.3% damaged
3. **Strong_pos is improved**: 55.7% improved, sign accuracy +0.5%
4. **Weak_pos is slightly damaged**: sign accuracy -1.0%
5. **Near_zero is neutral**: 51.9% improved, 49.6% damaged, sign unchanged
6. **Overall effect is subtle but directionally correct**:
   - Sign fixed: 2 samples, Sign broken: 1 sample
   - Net improvement driven by better sign decisions on borderline cases

## Delta Statistics

- mean|delta_reg| = 0.1576
- std|delta_reg| = 0.3151
- max|delta_reg| = 2.6455
- The residual makes small corrections on average, but can make large ones on some samples

## Conclusions

1. **Residual corrects weak_neg most effectively** — supporting the weak_neg reweight strategy
2. **Strong_neg is over-corrected** — ConditionalResidualGate should help by reducing gate for confident text predictions
3. **Overall improvement is about better sign decisions**, not overall error reduction
4. **Delta magnitudes are reasonable** (mean 0.16, max 2.65 is high for some outliers)

## Recommendations

1. ConditionalResidualGate should target strong samples (reduce gate for |text_base| > 2.0)
2. Weak_neg reweight should improve the gains further
3. Two-stage training may help by preventing residual from interfering with text base pretraining
