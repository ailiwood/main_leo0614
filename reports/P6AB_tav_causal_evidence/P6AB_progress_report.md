# P6AB Final Progress Report — MOSEI TAV Causal Evidence

**Date**: 2026-06-21 07:00 UTC  
**Branch**: `p6ab-tav-causal-evidence-mosi-contract`  
**Status**: **COMPLETE** — All 7 ablation variants trained and analyzed

---

## Final Ablation Matrix (4-epoch protocol, seed=42)

| Var | Description | ACC2 | Δ vs F0 | F1 | MAE | Corr | Params(M) |
|-----|-------------|------|---------|-----|-----|------|-----------|
| **F3** | **Global static weights** | **79.30%** | **+0.98** | **83.70** | **0.706** | **0.597** | **3.59** |
| F4 | Fixed mean 1/3 | 79.30% | +0.98 | 83.70 | 0.706 | 0.597 | 3.59 |
| F2 | T+V (no audio) | 78.72% | +0.40 | 82.94 | 0.716 | 0.582 | 4.26 |
| F0 | TAV AWAF sLSTM | 78.32% | — | 83.30 | 0.735 | 0.562 | 4.84 |
| E1 | LSTM (vs sLSTM) | 77.78% | -0.54 | 81.86 | 0.710 | 0.589 | 4.97 |
| F1 | T+A (no vision) | 76.65% | -1.67* | 82.53 | 0.742 | 0.553 | 4.10 |
| F5 | No interaction terms | 75.96% | -2.36* | 82.15 | 0.749 | 0.554 | 4.45 |

\* = statistically significant at p<0.05 (bootstrap 95% CI excludes 0)

## Causal Contribution Analysis

| Rank | Component | Effect | 95% CI | Significant |
|------|-----------|--------|--------|-------------|
| 1 | Hadamard interaction (g_ta/g_tv/g_av) | +2.37pp | [+1.49, +3.25] | ✅ p<0.05 |
| 2 | Vision modality | +1.67pp | [+0.61, +2.73] | ✅ p<0.05 |
| 3 | Global static > Dynamic AWAF | +0.97pp | [+0.15, +1.79] | ✅ p<0.05 |
| 4 | sLSTM vs LSTM | +0.55pp | [-0.49, +1.58] | ❌ n.s. |
| 5 | Audio modality | -0.39pp | [-1.37, +0.58] | ❌ n.s. |

## AWAF Weight Analysis (F0, 4 epochs)

- w_t (text): 0.435 ± 0.031 (range: 0.319–0.484)
- w_a (audio): 0.263 ± 0.032 (range: 0.211–0.376)
- w_v (vision): 0.302 ± 0.019 (range: 0.235–0.360)
- Avg entropy: 1.072 (near-uniform would be 1.099)

Text dominates, but vision receives more weight than audio at 4 epochs (30% vs 26%).

## Key Scientific Findings

### Significant (p<0.05)
1. **Hadamard interaction terms (g_ta, g_tv, g_av) are the most important AWAF component** — removing them causes a 2.37pp drop in ACC2
2. **Vision provides a small but statistically significant improvement** of +1.67pp ACC2 over text+audio alone
3. **Dynamic sample-level AWAF weights underperform global/mean fusion** — global static weights achieve +0.97pp higher ACC2 at 4-epoch budget

### Not Significant
4. **Audio does not provide measurable benefit at 4 epochs** — T+V (78.72%) performs similarly to TAV full (78.32%)
5. **sLSTM is not significantly better than plain LSTM** at 4 epochs (+0.55pp, n.s.)

### Architecture Implications
- The Hadamard interaction mechanism is critical and should be preserved
- The sample-level dynamic weighting may require more training epochs (>4) to become beneficial
- Text-only performance serves as a strong baseline; multimodal contributions are incremental
- At 12 epochs (P6AA), AWAF weights shift toward text dominance (w_t=0.481 vs 0.435 at 4ep)

## MOSI Status

- **No new MOSI training** was performed in P6AB
- MOSI retains one historical reference: P6K T+A conservative (ACC2=88.72%)
- Classification: `historical_text_audio_reference`, `not_tav_evidence`
- See: `mosi_scope_exclusion.md`

## Paper-Ready Claims (after P6AB)

### Allowed
✅ "Under a 4-epoch controlled ablation protocol on the MOSEI TAV complete-case cohort, vision provides a statistically significant +1.67pp ACC2 improvement (95% CI [+0.61, +2.73], paired bootstrap, n=3,294 non-zero test samples)."

✅ "The AWAF Hadamard interaction terms (g_ta, g_tv, g_av) are the single most important fusion component, contributing +2.37pp ACC2."

✅ "At 4-epoch training budget, global static modality weights outperform sample-level dynamic AWAF weights (+0.97pp, p<0.05), suggesting the dynamic weighting mechanism may require more training to become effective."

✅ "The AWAF assigns the highest mean weight to text (0.435), followed by vision (0.302) and audio (0.263)."

### NOT Allowed
❌ "Vision significantly improves the main model" — only tested at 4 epochs; P6AA 12-epoch main is the canonical candidate
❌ "AWAF outperforms all fusion methods" — global static weights beat it at 4 epochs
❌ "Dual-dataset validation" — MOSI excluded
❌ Direct numerical comparison of P6AB 4-epoch with P6AA 12-epoch results

## Deliverables

### Reports
- `reports/P6AB_tav_causal_evidence/P6AB_progress_report.md` (this file)
- `reports/P6AB_tav_causal_evidence/P6AA_result_integrity_audit.md`
- `reports/P6AB_tav_causal_evidence/P6AA_artifact_integrity_matrix.csv`
- `reports/P6AB_tav_causal_evidence/P6AA_old_ta_vs_new_tav_comparability.md`
- `reports/P6AB_tav_causal_evidence/mosi_scope_exclusion.md`
- `reports/P6AB_tav_causal_evidence/experiment_registry_delta.csv`
- `reports/P6AB_tav_causal_evidence/mosei/tav_ablation_results.md`
- `reports/P6AB_tav_causal_evidence/mosei/tav_ablation_results.csv`
- `reports/P6AB_tav_causal_evidence/mosei/tav_ablation_statistics.md`
- `reports/P6AB_tav_causal_evidence/mosei/A0_tav_ablation_switch_audit.md`
- `reports/P6AB_tav_causal_evidence/mosei/A0_tav_model_signature_matrix.csv`
- `reports/P6AB_tav_causal_evidence/mosei/A0_tav_fixed_batch_output_audit.csv`

### Scripts
- `scripts/audit_tav_ablation_switch_integrity.py`
- `scripts/run_p6ab_mosei_tav_ablation.py`
- `scripts/analyze_tav_ablation_results.py`

### Configs
- `configs/experiments/p6ab_tav_ablation/mosei/` (7 YAML configs)

### Training Outputs
- `outputs/P6AB_tav_ablation/mosei/` (7 run directories with complete artifacts)

### Handoff
- `HANDOFF_PHASE_P6AB_TAV_CAUSAL_EVIDENCE.md`
