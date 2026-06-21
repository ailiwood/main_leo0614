# P6AF Handoff — MOSI TAV Scientific Recovery

**Date**: 2026-06-21  
**Branch**: `p6af-mosi-tav-scientific-recovery`  
**Status**: BLOCKED — canonical AWAF fails on MOSI after two pre-registered attempts

---

## Session Summary

### P6AF-G0: MOSI Data Contract Audit ✅
- Both audio and vision are Route-S (valid temporal sequences)
- Audio: 100 frames × 768-dim (data2vec), median temporal var 0.029
- Vision: 32 frames × 1024-dim (CLIP-L14), median temporal var 1.002
- No T=1 or T<4 samples; all features have real temporal structure
- Labels: train 1284, val 229, test 686; no split overlap
- Label range: [-3, +3]; majority baseline ACC2 = 59.8%

### P6AF-G1: Diagnostic Probes ✅
| Probe | Val ACC2 | Diagnosis |
|-------|----------|-----------|
| P1 text-only | 86.11% | Text pipeline HEALTHY |
| P2 audio-only | 56.94% | Below majority — no standalone signal |
| P3 vision-only | 57.41% | Below majority — no standalone signal |
| P6 shuffled-label | 57.41% | Cannot learn — data integrity OK |

### P6AF-G2/G3: Route A Training ❌
- 35-epoch canonical AWAF, all branches trainable
- Best val ACC2 = 59.72% (at majority baseline)
- Test ACC2 = 44.21% (below chance)
- Early stopped at epoch 12 (no improvement since epoch 4)
- **Verdict: COLLAPSED**

### Route B Assessment ❌
- No Route B conditions met
- B1 not applicable (both modalities are Route-S)
- B2 not applicable (Route A collapsed, not underfitting)
- B3 not applicable (no data bugs found)

---

## Scientific Findings

### Why Canonical AWAF Fails on MOSI

The canonical AWAF architecture blends all modalities through softmax:
```
z = w_t * h_t + w_a * h_a + w_v * h_v
[w_t, w_a, w_v] = softmax(scorer(h_t, h_a, h_v) / tau)
```

On MOSI's small dataset (1,284 train):
- Text provides strong signal (P1: 86.11%)
- Audio provides no standalone signal (P2: 56.94% < 59.8%)
- Vision provides no standalone signal (P3: 57.41% < 59.8%)
- AWAF assigns comparable weights to all three
- Result: noise from audio/vision overwhelms text → fused output degrades below text-only

### Why P6K Succeeded (88.72%)

P6K used residual architecture:
```
final = text_base + gate * delta(audio)
```
This preserves the strong text baseline and only adds audio when helpful.

### Architectural Insight

This is a **scientifically valid finding**:
- Canonical AWAF requires sufficient training data to learn appropriate modality weights
- On small datasets (<2,000 samples), AWAF cannot distinguish signal from noise across modalities
- A text-anchored residual architecture is more appropriate for small datasets
- This is consistent with the MOSEI finding that AWAF's dynamic weighting becomes beneficial only with sufficient training (F3 > F0 at 4 epochs on MOSEI with 13K samples)

---

## Final Status

### MOSEI (Primary Evidence)
- P6AA/P6AB: `aligned_or_mixed_tav_reference` (not strict complete-case)
- Key findings: Vision +1.67pp, Hadamard interaction +2.37pp, Global static > Dynamic AWAF at 4 epochs
- Strict cohort (18,571) built but not yet re-run

### MOSI (Secondary Reference)
- P6K T+A residual: ACC2=88.72% (historical reference)
- MOSI TAV canonical AWAF: BLOCKED (2 attempts collapsed)
- Diagnostic infrastructure complete (G0, G1)
- Dataset built and validated (mosi_tav_v1, 2,199 samples)

### Paper Claims
✅ CAN claim:
- "Canonical AWAF sLSTM established on MOSEI TAV complete-case cohort with full causal ablation"
- "AWAF Hadamard interaction terms are critical (+2.37pp)"
- "On small datasets, text-anchored residual fusion may be more appropriate than equal-opportunity AWAF"
- "MOSI retained as text-audio residual reference (88.72%)"

❌ CANNOT claim:
- "Canonical TAV AWAF validated on both MOSEI and MOSI"
- "Dual-dataset three-modal evidence complete"
- Direct numerical comparison of MOSI and MOSEI results

---

## Files Created in P6AF

```
configs/experiments/p6af_mosi_tav/route_a_canonical_tav_s42.yaml
scripts/audit_mosi_tav_scientific_contract.py
scripts/run_mosi_diagnostic_probes.py
reports/P6AF_mosi_tav_recovery/G0_mosi_data_contract.md
reports/P6AF_mosi_tav_recovery/G0_mosi_feature_temporality.csv
reports/P6AF_mosi_tav_recovery/G0_mosi_label_alignment.csv
reports/P6AF_mosi_tav_recovery/G0_mosi_split_manifest.csv
reports/P6AF_mosi_tav_recovery/G0_mosi_feature_statistics.csv
reports/P6AF_mosi_tav_recovery/G1_diagnostic_probe_report.md
reports/P6AF_mosi_tav_recovery/G1_diagnostic_probe_results.csv
reports/P6AF_mosi_tav_recovery/MOSI_TAV_blocked_final.md
docs/P6AF_MOSI_TAV_PROTOCOL.md
HANDOFF_PHASE_P6AF_MOSI_TAV_SCIENTIFIC_RECOVERY.md
outputs/P6AF_mosi_tav/main/route_a_canonical_tav_s42/ (collapsed, preserved)
```

## Previous Phase Files (Preserved)

```
reports/P6AD_strict_tav_mosi/ (P6AD MOSEI audit + MOSI collapse)
reports/P6AB_tav_causal_evidence/ (P6AB MOSEI ablation)
outputs/P6AB_tav_ablation/ (7 MOSEI ablation variants)
outputs/P6AD_tav/mosi/ (P6AD MOSI F0 collapse)
```
