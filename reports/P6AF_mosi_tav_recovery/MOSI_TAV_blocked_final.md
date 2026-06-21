# MOSI TAV Scientific Recovery — Final Blocked Report

**Date**: 2026-06-21  
**MOSI_TAV_STATUS**: `blocked_after_two_attempts`  
**Attempts exhausted**: P6AD (4-epoch frozen), P6AF Route A (35-epoch full-train)

---

## Attempt History

### P6AD F0 (4-epoch, frozen text)
- Protocol: 4 epochs, freeze_text_base=true, all branches trainable
- Result: ACC2=42.23%, Corr=-0.03, val flat at 57.41%
- Verdict: Collapsed

### P6AF Route A (35-epoch, LoRA trainable, all branches trainable)
- Protocol: max 35 epochs, early_stop patience 8, min 10 epochs
- Text: RoBERTa-large base frozen, LoRA trainable from epoch 1
- Audio: data2vec 768d → sLSTM, trainable from epoch 1
- Vision: CLIP-L14 1024d → sLSTM, trainable from epoch 1
- Result: best val ACC2=59.72% at epoch 4, test ACC2=44.21%
- Early stopped at epoch 12 (no improvement since epoch 4)
- Verdict: Collapsed — never exceeded majority baseline

---

## Diagnostic Evidence (P6AF-G1)

| Probe | Val ACC2 | Verdict |
|-------|----------|---------|
| P0 majority baseline | 59.8% | Reference |
| P1 text-only (RoBERTa+LoRA) | **86.11%** | Text is healthy |
| P2 audio-only (data2vec+sLSTM) | 56.94% | Below majority |
| P3 vision-only (CLIP+sLSTM) | 57.41% | Below majority |
| P6 shuffled-label | 57.41% | Cannot learn (sanity OK) |

### Root Cause Analysis

1. **Text pipeline is healthy**: P1 achieves 86.11% — comparable to P6K T+A (88.72%)
2. **Audio has no standalone signal**: P2 at 56.94% is below majority
3. **Vision has no standalone signal**: P3 at 57.41% is below majority
4. **Canonical AWAF blends equal-opportunity**: All three modalities contribute to z through softmax weighting
5. **Result**: Noise from audio/vision overwhelms text signal → fused prediction degrades below text-only

### Why P6K Succeeded (88.72%)

P6K used **residual architecture**: `final = text_base + gate * delta(audio)`. This:
- Preserves strong text baseline (always contributes)
- Gate controls whether audio adds or not
- If audio is noisy, gate → 0, preserving text-only performance

Canonical AWAF lacks this text-preservation mechanism. All modalities compete equally through softmax.

---

## Route B Assessment

| Condition | Met? | Reason |
|-----------|------|--------|
| B1: Pooled features wrongly in sLSTM | ❌ No | G0 confirms both are Route-S |
| B2: Underfitting, text probe healthy | ❌ No | Route A collapsed (below majority), not underfitting |
| B3: Label/mask/metrics bug | ❌ No | G1 P6 shuffled cannot learn, confirming data integrity |

**No Route B conditions met. Protocol requires stop.**

---

## Recommendations

### For MOSI
1. Use **residual architecture** (text bypass + gate), not canonical AWAF
2. P6K demonstrates this works: ACC2=88.72%
3. The residual architecture can be described as "text-anchored multimodal fusion" in the paper

### For MOSEI
1. MOSEI's larger dataset (13,239 train) enables canonical AWAF to work
2. P6AB showed AWAF benefits (Hadamard interaction +2.37pp)
3. Continue using MOSEI as the primary three-modal evidence dataset

### Paper Strategy
```text
MOSEI (primary): Canonical T+A+V AWAF sLSTM — verified with full ablation
MOSI (secondary): Text-audio residual reference — demonstrates text-anchored fusion
Both: Feature versions documented; results NOT directly numerically compared
```

---

## Deliverables

- `reports/P6AF_mosi_tav_recovery/G0_mosi_data_contract.md`
- `reports/P6AF_mosi_tav_recovery/G1_diagnostic_probe_report.md`
- `reports/P6AF_mosi_tav_recovery/MOSI_TAV_blocked_final.md` (this file)
- `outputs/P6AF_mosi_tav/main/route_a_canonical_tav_s42/` (collapsed, preserved)
- `outputs/P6AD_tav/mosi/F0_full_tav_awaf_slstm_s42/` (collapsed, preserved)
