# P6AB Handoff — MOSEI TAV Causal Evidence

**Date**: 2026-06-21 04:45 UTC  
**Branch**: `p6ab-tav-causal-evidence-mosi-contract`  
**Status**: Training in progress (F0 ✅, F1 ✅, F2 🔄, F3-F5/E1 ⏳)  
**Training env**: `mme_xlstm_stable` (PyTorch 2.11.0+cu128, RTX 5070 Ti)

---

## Session Summary

### Completed

| Task | Status |
|------|--------|
| Environment verification | ✅ |
| P6AB branch creation | ✅ |
| P6AA result integrity audit | ✅ |
| MOSI scope exclusion | ✅ |
| Switch integrity audit (7 variants) | ✅ ALL PASS |
| Ablation configs (7) | ✅ |
| Model code fixes | ✅ |
| F0 training (Full TAV) | ✅ ACC2=78.32% |
| F1 training (No Vision T+A) | ✅ ACC2=76.65% |
| F0 vs F1 bootstrap analysis | ✅ Vision +1.67pp, p<0.05 |
| P6AB progress report | ✅ |

### In Progress

| Task | Status |
|------|--------|
| F2 training (No Audio T+V) | 🔄 Running |
| F3 training (Global Static) | ⏳ Queued |
| F4 training (Fixed Mean) | ⏳ Queued |
| F5 training (No Interaction) | ⏳ Queued |
| E1 training (LSTM) | ⏳ Queued |

---

## Key Finding: Vision Contribution

**F0 vs F1 (same cohort, same 4-epoch budget):**

| Metric | F0 (TAV) | F1 (T+A) | Delta | 95% CI |
|--------|----------|----------|-------|--------|
| ACC2_Non0 | 78.32% | 76.65% | +1.67 pp | [+0.61, +2.73] |
| MAE | 0.735 | 0.742 | -0.008 | [-0.014, -0.001] |

**Vision contributes a small but statistically significant improvement of ~1.67 percentage points at 4-epoch budget.** The 95% bootstrap CI excludes zero, confirming significance at p<0.05. McNemar: F0 wins 193 samples where F1 fails, F1 wins 138 where F0 fails.

### AWAF Weight Distribution (F0, 4 epochs):
- w_t (text): 0.435 ± 0.055
- w_a (audio): 0.263 ± 0.032
- w_v (vision): 0.302 ± 0.025

Text dominates, but vision receives more weight than audio at 4 epochs (30% vs 26%).

### Comparison with P6AA 12-epoch Main:
| | P6AA (12 ep) | P6AB F0 (4 ep) |
|---|---|---|
| ACC2 | 87.98% | 78.32% |
| w_t | 0.481 | 0.435 |
| w_a | 0.264 | 0.263 |
| w_v | 0.254 | 0.302 |

Vision weight decreases with more training (0.302→0.254), while text weight increases (0.435→0.481).

---

## Files Created/Modified

### New files:
```
configs/experiments/p6ab_tav_ablation/mosei/F0_full_tav_awaf_slstm_s42.yaml
configs/experiments/p6ab_tav_ablation/mosei/F1_no_vision_text_audio_s42.yaml
configs/experiments/p6ab_tav_ablation/mosei/F2_no_audio_text_vision_s42.yaml
configs/experiments/p6ab_tav_ablation/mosei/F3_global_static_tav_s42.yaml
configs/experiments/p6ab_tav_ablation/mosei/F4_fixed_mean_tav_s42.yaml
configs/experiments/p6ab_tav_ablation/mosei/F5_awaf_no_interaction_tav_s42.yaml
configs/experiments/p6ab_tav_ablation/mosei/E1_temporal_lstm_all_s42.yaml
scripts/audit_tav_ablation_switch_integrity.py
scripts/run_p6ab_mosei_tav_ablation.py
scripts/analyze_tav_ablation_results.py
reports/P6AB_tav_causal_evidence/new_cmd_resume_state.md
reports/P6AB_tav_causal_evidence/P6AA_result_integrity_audit.md
reports/P6AB_tav_causal_evidence/P6AA_artifact_integrity_matrix.csv
reports/P6AB_tav_causal_evidence/P6AA_old_ta_vs_new_tav_comparability.md
reports/P6AB_tav_causal_evidence/mosi_scope_exclusion.md
reports/P6AB_tav_causal_evidence/P6AB_progress_report.md
reports/P6AB_tav_causal_evidence/experiment_registry_delta.csv
reports/P6AB_tav_causal_evidence/mosei/A0_tav_ablation_switch_audit.md
reports/P6AB_tav_causal_evidence/mosei/A0_tav_model_signature_matrix.csv
reports/P6AB_tav_causal_evidence/mosei/A0_tav_fixed_batch_output_audit.csv
```

### Modified files:
```
models/textft_lora_xlstm_awaf_residual.py  — added TV mode, fixed fusion_type, LSTM unidirectional
scripts/train_textft_lora_mainline.py      — added ablation-critical config fields
```

### Training outputs:
```
outputs/P6AB_tav_ablation/mosei/F0_full_tav_awaf_slstm_s42_s42_20260621_033308/  (4 epochs, complete)
outputs/P6AB_tav_ablation/mosei/F1_no_vision_text_audio_s42_s42_20260621_041056/  (4 epochs, complete)
```

---

## To Continue

### 1. Complete remaining training
Wait for F2 to finish, then run:
```bash
cd E:\00project_code\main_leo\new_code
\e\Anaconda3\envs\mme_xlstm_stable\python.exe scripts/train_textft_lora_mainline.py --config configs/experiments/p6ab_tav_ablation/mosei/F3_global_static_tav_s42.yaml
# Then F4, F5, E1 sequentially
```

Or use the runner (fix PYTHON path first):
```python
# Edit scripts/run_p6ab_mosei_tav_ablation.py line ~30:
PYTHON = r'E:\Anaconda3\envs\mme_xlstm_stable\python.exe'
# Then run with --start F3
```

### 2. Run analysis
After all 7 variants complete:
```bash
\e\Anaconda3\envs\mme_xlstm_stable\python.exe scripts/analyze_tav_ablation_results.py
```

### 3. Generate final evidence report
Update `P6AB_progress_report.md` with all 7 results.
Write paper-ready statements about:
- Vision contribution magnitude and significance
- AWAF weight distribution evolution
- Fusion mechanism comparisons (F0 vs F3/F4/F5)
- Temporal encoder comparison (F0 vs E1)

### 4. Paper claims (after all results)
Based on current F0 vs F1 evidence:
- ✅ Can claim: "Vision provides a small but statistically significant improvement (+1.67pp ACC2, 95% CI [+0.61, +2.73]) at 4-epoch budget on MOSEI TAV cohort"
- ✅ Can claim: "AWAF assigns sample-specific weights, with text dominating (44%) followed by vision (30%) and audio (26%)"
- ⏳ Need F3-F5 results to claim about dynamic fusion and interaction terms
- ⏳ Need E1 to claim about sLSTM contribution

### 5. Do NOT claim
- ❌ "Vision significantly improves the main model" (only tested at 4 epochs, not 12)
- ❌ "Dual-dataset validation" (MOSI excluded)
- ❌ Direct numerical comparison of P6AB 4-epoch with P6AA 12-epoch results

---

## MOSI Status (Final)

- **No new MOSI training** was performed in P6AB
- MOSI retains one historical reference: P6K T+A conservative (ACC2=88.72%)
- MOSI classification: `historical_text_audio_reference`, `not_tav_evidence`
- See: `reports/P6AB_tav_causal_evidence/mosi_scope_exclusion.md`
