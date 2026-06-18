# P6J Val/Test Gap Audit

## Split Sizes

| Split | N | % |
|---|---|---|
| Train | 1284 | 58.4% |
| Val | 229 | 10.4% |
| Test | 686 | 31.2% |

Val set (229 samples) is only 10.4% of total.

## Gap History

| Run | val_ACC2 | test_ACC2 | gap |
|---|---|---|---|
| P6H original | 87.50% | 86.43% | 1.07% |
| P6J textbase | 86.57% | 83.99% | **2.58%** |
| P6I text_conf | 87.50% | 85.06% | **2.44%** |

Gap increased from 1.07% (P6H) to 2.44-2.58% (P6I/P6J).

## Possible Causes

1. **Val set too small (229):** High variance in val metric, best epoch unreliable
2. **Early stopping on val:** Selects checkpoint that overfits val characteristics
3. **Model architecture change:** Class-based vs inline may affect random init
4. **Seed sensitivity:** Single seed comparison may not capture normal variance

## Recommendations

1. ✅ Final paper uses official split (do not change)
2. ✅ Test never used for tuning
3. 🟡 Use val for model selection only, not hyperparameter optimization
4. 🟡 Consider reporting mean±std over 3 seeds for final results

## Conclusion

- Val/test gap is larger in P6I/P6J than P6H original
- Likely due to early stopping on tiny val set + single seed
- Official split is correct, no change needed
- Multi-seed averaging would provide more reliable estimates
