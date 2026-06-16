# P4R Split & Early-Stop Audit

## Critical Finding: TEST LEAKAGE

The P4A training loop (`p4a_mosi_c0_formal.py`) evaluates on TEST set every epoch and selects best_epoch by test ACC2_Non0:

```python
er = trainer.evaluate(te_ld)  # te_ld = TEST loader
if er['metrics']['ACC2_Non0'] > best_acc:
    best_acc = er['metrics']['ACC2_Non0']
    best_ep = ep
```

**This is test leakage.** The model selection (best_epoch, best_checkpoint) used data from the test set.

## Impact

| Seed | Best Ep (by test) | Best ACC2 | Final Ep 60 ACC2 | Gap |
|:--:|:--:|:--:|:--:|:--:|
| 42 | 20 | 76.68% | 75.00% | +1.68% |
| 2024 | 26 | 76.83% | 75.30% | +1.53% |
| 1234 | 21 | 76.83% | 75.46% | +1.37% |

The best_epoch values give inflated results because the model was selected by peeking at test data.

## Correct Protocol

For paper-valid results:
1. Split MOSI train into train'/val' (e.g., 80/20)
2. Use val' for epoch selection and early stopping
3. Evaluate on test only ONCE at the end
4. Report test metrics at val'-selected best epoch

OR use the official MOSI val split (229 samples) for model selection.

## P4A Results Status

**P4A results are NOT directly paper-usable** due to test leakage.
They can be used as internal development reference but must be re-run with strict protocol for paper tables.
