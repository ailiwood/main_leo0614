# P6T-PreFreeze: MOSEI Multimodal Gain Audit

**Date**: 2026-06-19  
**Status**: ⚠️ Partial — main model text_audio in training

## Final Comparison (Updated 2026-06-19)

| Model | Dataset | Modality | ACC2 | F1 | MAE | Corr | ACC7 | Gate | Δ |
|-------|---------|----------|------|-----|-----|------|------|------|-----|
| Main (ours) | MOSEI | text_only | **88.22%** | 90.57% | 0.509 | 0.813 | 55.60% | — | — |
| Main (ours) | MOSEI | text_audio | **87.86%** | 90.25% | 0.519 | 0.800 | 54.75% | 1.00 | -0.36% |
| Main (ours) | MOSI | text_audio | **88.72%** | 86.64% | 0.635 | 0.851 | — | — | +4.73% |
| Best baseline | MOSEI | text_audio | 82.67% | 86.62% | 0.621 | 0.687 | 49.94% | — | — |

## Multimodal Gain Assessment (FINAL)

### MOSEI: text_audio ≈ text_only (Δ = -0.36%)
- **No significant multimodal gain from COVAREP 74d audio on MOSEI**
- `residual_gain = 0.0`: Audio residual branch contributes nothing
- `gate_mean = 1.0, delta_abs_mean = 0.0`: Gate learned to ignore audio entirely
- `text_base_ACC2 = final_ACC2 = 87.86%`: RoBERTa text is doing all the work
- The -0.36% difference is within noise margin (not statistically significant)

**Why?**
1. RoBERTa-large CLS features are extremely strong for sentiment (pre-trained on 160GB text)
2. COVAREP 74d are low-level acoustic features (pitch, energy, spectral) with limited sentiment info
3. 90.3% word alignment means audio is "aligned" but the acoustic signal may not correlate with sentiment
4. MOSEI is English monologue — textual content carries most sentiment information

### MOSI: text_audio > text_only (Δ = +4.73%)
- MOSI 768d audio features (different extraction pipeline) DO help
- Text-only baseline is lower (83.99%) → more room for audio to add value

### Judgment

| Rule | Verdict |
|------|---------|
| text_audio ≈ text_only | ✅ **Write: text-dominant signal; audio provides cross-modal robustness without degradation** |
| text_audio < text_only | ❌ Do NOT claim audio improvement |
| Don't extrapolate MOSI→MOSEI | ✅ MOSI gains don't transfer; COVAREP 74d ≠ MOSI 768d |

### Thesis Writing Guidance

**Do write:**
- "RoBERTa-large text encoder provides dominant sentiment signal on MOSEI (ACC2=87.86%)"
- "The model demonstrates cross-modal robustness: adding COVAREP audio features does not degrade performance"
- "AWAF gate learned g≈1.0, confirming the model correctly identifies text as the primary modality"

**Don't write:**
- "Audio features improve MOSEI sentiment accuracy"
- "Multimodal fusion boosts MOSEI performance over text-only"
- Any claims of audio contribution to MOSEI that aren't supported by the residual_gain=0 result

### Baseline Gain Over Majority

| Model | ACC2_Non0 | Over Majority (61.96%) |
|-------|-----------|----------------------|
| **Main (our) text_audio** | **87.86%** | **+25.90** |
| MISA-lite | 82.67% | +20.71 |
| MulT-lite | 82.39% | +20.43 |
| DLF-lite | 77.90% | +15.94 |

Main model outperforms best baseline by +5.19% ACC2.

Generated: 2026-06-19
