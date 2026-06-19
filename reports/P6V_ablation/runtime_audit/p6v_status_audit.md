# P6V-R: Full Status Audit

**Time**: 2026-06-20 02:20 UTC+8 | **Active python.exe**: 3

## MOSEI (6 target)

| Model | Status | Ep | ACC2 | F1 | MAE | Action |
|-------|--------|----|------|-----|-----|--------|
| main_awaf_slstm | INHERITED | 12 | 87.86% | 90.25% | 0.519 | ✅ Done |
| fusion_gated | FORMAL | 4 | 76.02% | 83.08% | 0.746 | ✅ Done |
| awaf_no_interaction | RUNNING | 2/4 | — | — | — | ⏳ Wait |
| fusion_mean | SMOKE | 1 | 38.04% | 0 | 0.979 | 🔄 Need formal |
| encoder_gru | PENDING | — | — | — | — | 🔄 Need formal |
| encoder_no_temporal | PENDING | — | — | — | — | 🔄 Need formal |

## MOSI (10 target)

| Model | Status | Ep | ACC2 | Action |
|-------|--------|----|------|--------|
| main_awaf_slstm | INHERITED | 20 | 88.72% | ✅ Done |
| awaf_no_context | COLLAPSED | 5 | 42.23% | ❌ All-positive collapse |
| awaf_no_interaction | COLLAPSED | 5 | 42.23% | ❌ All-positive collapse |
| encoder_gru | COLLAPSED | 5 | 42.23% | ❌ All-positive collapse |
| encoder_lstm | COLLAPSED | 5 | 42.23% | ❌ All-positive collapse |
| encoder_no_temporal | COLLAPSED | 5 | 42.23% | ❌ All-positive collapse |
| fusion_concat | COLLAPSED | 5 | 42.23% | ❌ All-positive collapse |
| fusion_fixed | COLLAPSED | 5 | 42.23% | ❌ All-positive collapse |
| fusion_gated | COLLAPSED | 5 | 42.23% | ❌ All-positive collapse |
| fusion_mean | SMOKE/PENDING | 1 | 57.77% | ❌ Smoke only |

## MOSI Collapse Analysis

All 8 MOSI formal attempts collapsed identically:
- val_ACC2=57.41% = MOSI val pos/non0 (all-positive prediction)
- test ACC2=42.23% = MOSI test pos/non0
- train_loss: 1.51→1.48 (barely decreasing)
- gate_mean=1.0, delta=0 (no residual contribution)
- 5 epochs, early_stopped at epoch 1

**Root cause hypotheses**:
1. MOSI too small (1284 samples) for RoBERTa+LoRA text_audio mode with current config
2. AWAF-always-created change introduced training dynamics issue
3. 8 epoch/patience=4 insufficient for convergence
4. P6K succeeded with 20 epoch/patience=6/different code version

**Recommended**: Re-run MOSI ablation with increased epochs (12-20) or fix root cause first.

## Completed Formal Results (for thesis)

| Dataset | Model | ACC2 | Δ vs Main |
|---------|-------|------|-----------|
| MOSEI | main_awaf_slstm | 87.86% | — |
| MOSEI | fusion_gated | 76.02% | -11.84% |

Generated: 2026-06-20
