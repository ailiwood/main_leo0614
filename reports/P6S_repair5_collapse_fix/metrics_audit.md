# P6S-Repair-5: Metrics Audit Report

## Audit Scope
Full audit of `utils/metrics.py` covering 7 metrics: MAE, Corr, ACC2_Non0, F1_Non0, ACC2_Has0, F1_Has0, ACC7.

## Findings

### Critical: NaN Silent Pass-Through
**Status**: FIXED

`compute_mae()` and `compute_all_metrics()` previously accepted NaN predictions silently:
- MAE: If all preds NaN, returned NaN (no warning)
- Corr: If all preds NaN, returned 0.0 (no warning)
- ACC2: NaN preds treated as negative → incorrect results

**Fix**: 
- `compute_mae`: Now raises ValueError if ALL predictions are NaN
- `compute_all_metrics`: Now raises ValueError if >10% of predictions are NaN
- Both provide informative error messages with NaN counts

### Pass: ACC2_Non0 Filtering
- Correctly excludes label == 0 using `nz = targets != 0`
- Correctly computes accuracy on Non0 subset
- Returns 0.0 when all labels are 0 (no crash)

### Pass: ACC2_Has0
- Includes all labels (including 0)
- Uses same threshold logic as Non0

### Pass: F1 Computation
- F1_Non0: Correctly excludes zero labels
- F1_Has0: Includes all labels
- Edge cases: zero denominator → returns 0.0
- All-negative predictions → F1=0 (correct, no positive predictions)

### Pass: ACC7
- Correctly rounds to [-3, +3] with 7 classes
- NaN predictions → valid count = 0 → returns 0.0

### Pass: Corr Edge Cases
- Constant predictions → std=0 → returns 0.0
- NaN predictions → valid=[] → len < 2 → returns 0.0
- No divide-by-zero risks

### Pass: Threshold Auto-Detection
- Auto-detects probability vs logit format
- If all values in [0,1] → threshold=0.5
- Otherwise → uses provided threshold (default 0.0)

## Unit Tests (tests/test_metrics_mosei_non0.py)
All 8 tests passed:
1. Normal pos/neg labels ✓
2. Labels containing 0 ✓
3. All-positive predictions ✓
4. All-negative predictions ✓
5. NaN prediction fail-fast ✓
6. Empty Non0 returns 0 ✓
7. MOSEI majority simulation ✓
8. ACC7 computation ✓

## MOSEI-Specific Verification

### Why ACC2=38.04% Is a Constant-Prediction Artifact
```
Test set: 4221 samples (2041 pos, 1253 neg, 927 zero)
Non0 samples: 3294 (2041 pos, 1253 neg)

If model predicts ALL NEGATIVE:
  acc2_non0 = neg_correct / non0 = 1253/3294 = 38.0389%
  
If model predicts ALL POSITIVE:
  acc2_non0 = pos_correct / non0 = 2041/3294 = 61.9611%
```

The 38.0389% means ALL models predict all-negative — exactly matching neg_non0/non0.

### Metric Validation Matrix
| Scenario | ACC2_Non0 | F1_Non0 | MAE | Corr | Expected |
|----------|-----------|---------|-----|------|----------|
| All NaN preds | 38.04% | 0 | NaN | 0 | FAIL-FAST |
| All constant neg | 38.04% | 0 | >0 | 0 | Valid |
| All constant pos | 61.96% | >0 | >0 | 0 | Valid |
| Random preds | ~50% | ~50% | >0 | ~0 | Valid |
| Good preds | >60% | >60% | <1.0 | >0.3 | Valid |

## Recommendations
1. ✅ NaN fail-fast added to compute_all_metrics
2. ✅ NaN fail-fast added to compute_mae
3. ✅ Unit tests cover all edge cases
4. Future: Consider adding prediction distribution logging (mean/std/pos_ratio per epoch)

Generated: 2026-06-19
