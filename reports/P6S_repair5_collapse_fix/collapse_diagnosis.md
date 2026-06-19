# P6S-Repair-5: Collapse Diagnosis Report

## Executive Summary

**Root Cause**: COVAREP audio features contain `-inf` values (from `log(0)` in feature extraction). When these pass through `Linear → ReLU → GRU`, they produce NaN that propagates through the entire model, causing all 8 baseline models to output NaN regression predictions, which map to constant all-negative binary predictions.

## Diagnosis Chain

### 1. Label & Split Analysis
- MOSEI test set: 4221 samples (2041 pos, 1253 neg, 927 zero)
- neg_non0/non0 = 1253/3294 = 38.0389%
- ALL 8 models: ACC2_Non0 = 38.0389% (matches exactly)
- ALL 8 models: F1_Non0 = 0.0
- Conclusion: All models predict ALL NEGATIVE on every sample

### 2. Prediction Analysis
- predictions_test.csv: prediction column ALL NaN
- `torch.where(NaN >= 0, 1.0, -1.0)` → ALL -1.0 (NaN >= 0 is False)
- All 8 models produce identical NaN predictions
- Conclusion: NaN in model regression output → constant binary classification

### 3. Metrics Audit
- `compute_mae`: all preds NaN → returns NaN ✓
- `compute_corr`: all preds NaN, valid=[] → std=0 → returns 0.0 ✓
- `compute_acc2_non0`: all-negative predictions → 38.0389% ✓
- `compute_f1_non0`: no positive predictions → 0.0 ✓
- Old code: no NaN fail-fast → silent corruption

### 4. RoBERTa Cache Check
- train: (14700, 1024) mean=-0.032, non-zero ✓
- valid: (1759, 1024) mean=-0.032, non-zero ✓
- test: (4221, 1024) mean=-0.032, non-zero ✓
- Cache complete, correct, not the issue

### 5. Audio Feature Analysis
- Feature files: audio shape [100, 74] COVAREP ✓
- Sample check: -inf found in ~2-5% of files
- Example: `-3nNcZdcdvU_0_0.npz`: 2/7400 values are -inf
- train: 24/1000 sampled files have -inf
- valid: 49/1000 sampled files have -inf
- test: 24/1000 sampled files have -inf

### 6. NaN Propagation Chain
```
COVAREP audio (-inf) → collate (no cleaning) → batch['audio'] has -inf
→ Linear(74→128)(-inf) → -inf*weight+0*weight → NaN
→ ReLU(NaN) → NaN
→ GRU(NaN) → NaN hidden states
→ Cross-attention(NaN) → NaN
→ Head(NaN) → NaN regression output
→ torch.where(NaN>=0, 1.0, -1.0) → ALL -1.0
→ ACC2_Non0 = neg_non0/non0 = 38.0389%
```

### 7. Why All Models Were Identical
Different architectures (MulT, TFN, LMF, MISA, SelfMM, MMIM, MLCL, DLF) all:
1. Receive the same NaN audio features
2. All produce NaN at the first Linear layer
3. NaN propagates identically regardless of architecture
4. All produce identical all-negative binary predictions

### 8. Why Train Loss Was NaN
- Loss = L1Loss(NaN_reg, real_labels) = NaN
- metrics_epoch.csv shows train_loss=nan for all 5 epochs
- Gradient of NaN = NaN (backprop fails silently)
- Weights don't actually change meaningfully

## Verification After Fix

| Metric | Before Fix (all 8 models) | After Fix (MulT-lite) |
|--------|--------------------------|----------------------|
| ACC2_Non0 | 38.0389% (exact constant) | 63.36% |
| F1_Non0 | 0.0 | 75.41% |
| ACC2_Has0 | 29.6849% | 44.56% |
| MAE | NaN | 0.8591 |
| Corr | 0.0 | 0.2808 |
| ACC7 | 0.0 | 39.73% |
| Train Loss | NaN | 0.31-0.59 |
| Prediction std | 0 (all constant) | >0 (varies) |

## Fix Applied

1. **`data/textft_multimodal_dataset.py`**: Added `_clean_features()` to replace -inf/+inf with 0.0 in collate function
2. **`utils/metrics.py`**: Added NaN detection (>10% NaN → ValueError fail-fast)
3. **`scripts/train_baseline_lite.py`**: Added collapse guard with epoch-level checks

Generated: 2026-06-19
