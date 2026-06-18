# P6J State Confirmation

## File Status

| File | Status |
|---|---|
| HANDOFF_P6H_REPAIR.md | ✅ |
| P6I_stage_decision.md | ✅ |
| P6I_text_conf_residual_result.md | ⚠️ Missing (created from result.json) |
| P6I_text_conf_predictions_test.csv | ✅ (copied from outputs) |
| P6I_av_quality_matrix.csv | ✅ |
| train_textft_lora_mainline.py | ✅ |
| mosi_text_conf_residual.yaml | ✅ |
| mosi_textbase_recovery.yaml | ✅ |

## Confirmed Facts

| Fact | Value |
|---|---|
| P6I text_conf completed | ✅ 20 epochs, best E12 |
| text_conf text_base test ACC2 | 85.06% |
| text_conf final test ACC2 | 85.06% |
| text_conf residual_gain | 0.00% |
| text_conf gate_mean | 0.75 (OPEN) |
| text_conf delta_abs_mean | 0.23 (LARGE) |
| text_conf AWAF weights (t/a/v) | 0.30/0.29/0.42 |
| P6H original text_base test ACC2 | 86.43% |
| P6I textbase recovery val | 86.57% |
| P6I textbase recovery test | 83.99% |

## Verdict

1. ✅ gate opened, delta enlarged — **architecture fix works**
2. ❌ text_base 85.06% ≠ P6H original 86.43% — **regression exists**
3. ❌ residual_gain still 0.00% — **delta direction suspect**
4. ✅ Need to first reproduce P6H original text_base, then audit delta direction
