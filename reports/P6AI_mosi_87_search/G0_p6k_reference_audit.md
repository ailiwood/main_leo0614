# P6AI-G0: P6K 88.72% Reference Audit

**Date**: 2026-06-21

## P6K Result Summary

| Metric | Value |
|--------|-------|
| Mode | `text_audio_residual` |
| ACC2_Non0 | 88.72% |
| F1_Non0 | 86.64% |
| MAE | 0.635 |
| Corr | 0.851 |
| ACC7 | 46.65% |
| Trainable params | 3.92M |
| Best epoch | 12 (of 50) |
| Best val ACC2 | 87.50% |
| Gate mean | 0.416 |
| Delta abs mean | 0.064 |

## Architecture

- Text: RoBERTa-large + LoRA + TextMLP → text_base
- Audio: data2vec 768d → sLSTM → pooling → audio correction
- Fusion: `final = text_base + gate * delta` (residual with uncertainty gate)
- Gate initialized at bias=2.0 → sigmoid(2.0) ≈ 0.88 at init
- **NOT canonical AWAF** — no softmax weighting, no forced blending

## Feature Version

P6K used `features_strong_sequence_mosi_v3_T40`:
- audio: (100, 768) data2vec
- vision: (40, 768) — present but NOT used (text_audio mode)
- text: pre-encoded (8, 1024) — but dataset uses raw text from CSV

Current `mosi_tav_v1`:
- audio: (100, 768) data2vec — SAME
- vision: (32, 1024) CLIP-L14 — DIFFERENT (1024d vs 768d)

## Classification

**p6k_reference_real_but_not_directly_reproducible** (Category B)

Reasons:
- P6K uses different feature root (v3_T40 vs mosi_tav_v1)
- P6K uses text_audio mode (no vision) — not valid as TAV evidence
- P6K uses residual architecture — not canonical AWAF
- No predictions_test.csv found for independent re-computation
- Checkpoint exists but uses old feature root

## What Can Be Learned

1. **Text-anchored residual works on MOSI**: The architecture preserves text baseline and only adds audio via gate
2. **Gate learns moderate trust**: gate=0.42 means ~42% trust in audio correction
3. **Small corrections suffice**: delta_abs=0.064 — fine-grained adjustments
4. **20-25 epochs enough**: Best epoch at 12 of 50
5. **Moderate batch effective**: batch=4, accum=16 → effective batch=64

## What Cannot Be Used

- P6K score (88.72%) cannot be written as TAV or AWAF result
- P6K cannot serve as TAV control group
- P6K config cannot be directly loaded for TAV training
- P6K text+audio result cannot be compared numerically with TAV results
