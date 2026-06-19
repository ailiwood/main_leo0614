# P6T-PreFreeze: P6S-Repair-5 Fact Check Report

**Date**: 2026-06-19  
**Status**: ✅ Verified

## 1. 8 MOSEI Baseline Output Integrity

| # | Model | best_model.pth | predictions_test.csv | result.json | metrics_best.json | ACC2 | Status |
|---|-------|:---:|:---:|:---:|:---:|-------|--------|
| 1 | MulT-lite | ✅ | ✅ | ✅ | ✅ | 82.39% | ✅ |
| 2 | SelfMM-lite | ✅ | ✅ | ✅ | ✅ | 82.60% | ✅ |
| 3 | TFN-lite | ✅ | ✅ | ✅ | ✅ | 81.48% | ✅ |
| 4 | LMF-lite | ✅ | ✅ | ✅ | ✅ | 81.85% | ✅ |
| 5 | MISA-lite | ✅ | ✅ | ✅ | ✅ | 82.67% | ✅ |
| 6 | MMIM-lite | ✅ | ✅ | ✅ | ✅ | 82.67% | ✅ |
| 7 | MLCL-lite | ✅ | ✅ | ✅ | ✅ | 80.66% | ✅ |
| 8 | DLF-lite | ✅ | ✅ | ✅ | ✅ | 77.90% | ✅ |

**Note**: train.log missing for all 8 (output was captured in background task log). All other outputs present.

## 2. P6S-Repair-4 Collapse Isolation

| Item | Status |
|------|--------|
| 12 runs moved to `outputs/_invalid/` | ✅ |
| invalid_outputs_manifest.csv exists | ✅ |
| Registry marked invalid_collapse/D | ✅ |

## 3. Code Fixes Verification

| Fix | File | Exists | Tested |
|-----|------|:---:|:---:|
| `_clean_features()` in collate_textft | data/textft_multimodal_dataset.py | ✅ | ✅ 50 batches clean |
| NaN fail-fast in compute_mae | utils/metrics.py | ✅ | ✅ unit test |
| NaN fail-fast in compute_all_metrics | utils/metrics.py | ✅ | ✅ unit test |
| Collapse guard in train_baseline_lite | scripts/train_baseline_lite.py | ✅ | ✅ smoke verified |
| 8 unit tests | tests/test_metrics_mosei_non0.py | ✅ | ✅ 8/8 pass |

## 4. Main Model MOSEI text_audio Status

| Phase | Status |
|-------|--------|
| P6S_repair2 (old) | ❌ Empty outputs — never completed |
| P6T_prefreeze (current) | 🔄 Training in background (12 epochs, ~84 min ETA) |

## 5. MISA/MMIM Duplicate

| Check | Finding |
|-------|---------|
| Code identical? | ✅ Yes (byte-for-byte except class name) |
| Configs identical? | ✅ Yes |
| Predictions identical? | ✅ Yes (4221/4221 match) |
| Action | MMIM-lite excluded from main table |

## 6. MOSI Baseline Collapse Review

| Phase | Status | ACC2 | F1 | Pred Health |
|-------|--------|------|-----|-------------|
| P6N | ❌ Collapsed | 42-58% | 0 | Constant |
| P6O | ❌ Collapsed | 42-50% | ~55% | Constant |
| P6P rescue | ✅ Healthy | 77% | 72-73% | HEALTHY |
| P6Q rescue | ✅ Healthy | 76-78% | 71-72% | HEALTHY |
| P6R 80pass | ✅ Healthy | 75-77% | 70-72% | HEALTHY |

MOSI audio features (768d) do NOT have -inf issue. The P6N/P6O collapse was from training configuration issues, not data corruption. P6P/Q/R rescue already resolved.

## 7. Global Fix Coverage

| Training Script | Uses collate_textft? | Covered by fix? |
|-----------------|:---:|:---:|
| train_baseline_lite.py | ✅ | ✅ |
| train_textft_lora_mainline.py | ✅ | ✅ |
| train_textft_xlstm_awaf_residual.py | ✅ | ✅ |
| eval_checkpoint.py | ✅ | ✅ |
| validate_mosei_fast_interfaces.py | ✅ | ✅ |

## Verdict
All facts confirmed. Proceed to P6T execution.

Generated: 2026-06-19
