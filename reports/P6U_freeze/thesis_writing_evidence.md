# P6U Freeze: Thesis Writing Evidence Summary

**Date**: 2026-06-20 | **Status**: FROZEN for thesis Ch3-5

## 1. Results That CAN Be Written Into Thesis

### Main Model Results (Table 4.x)
| Dataset | Modality | ACC2_Non0 | F1_Non0 | MAE | Corr |
|---------|----------|-----------|---------|-----|------|
| MOSI | text_audio | **88.72%** | 86.64% | 0.635 | 0.851 |
| MOSEI | text_audio | **87.86%** | 90.25% | 0.519 | 0.800 |

### Baseline Results (Table 4.x)
| Dataset | Best Baseline | ACC2 | Main Δ |
|---------|--------------|------|--------|
| MOSI | TFN-lite 77.59% | — | +11.13% |
| MOSEI | MISA-lite 82.67% | — | +5.19% |

### Multimodal Gain
| Dataset | text_only | text_audio | Δ | Interpretation |
|---------|-----------|------------|------|----------------|
| MOSI | 83.99% | 88.72% | +4.73% | Audio helps |
| MOSEI | 88.22% | 87.86% | -0.36% | Text-dominant |

## 2. What CANNOT Be Written

- ❌ "MOSEI audio features improve accuracy" (residual_gain=0)
- ❌ "MMIM-lite achieves 82.67%" (identical to MISA-lite)
- ❌ "Vision modality tested on MOSEI" (all-zero features)
- ❌ "Baseline-lite is official reimplementation" (lite versions, not official)
- ❌ Any smoke/subset/collapse result as formal result
- ❌ MOSEI text_only as final main model

## 3. Chapter 3 (Method) — Writeable Content

1. **System architecture**: TextFTLoRAXLSTMAWAFResidual
   - RoBERTa-large + LoRA for text encoding
   - sLSTM (1-layer) for temporal encoding of audio/vision
   - AWAF (Adaptive Weighted Attention Fusion) for sample-level modality weighting
   - Residual gate mechanism for text-base + multimodal residual
   - Code: `models/textft_lora_xlstm_awaf_residual.py`

2. **AWAF module**: `models/fusion/awaf.py`
   - Cross-modal context enhancement
   - First-order + second-order Hadamard interaction weights
   - w_t + w_a + w_v = 1, sample-level interpretable weights
   - Tau temperature parameter for softmax sharpness

3. **sLSTM encoder**: `models/encoders/slstm.py`
   - Exponential gating with stabilizer states
   - Mask-aware sequence processing

## 4. Chapter 4 (Experiments) — Writeable Content

1. **Datasets**: MOSI (1284 samples), MOSEI (20680 samples)
2. **Features**: RoBERTa-large CLS + LoRA (text), COVAREP 74d (MOSEI audio), 768d (MOSI audio)
3. **Setup**: AdamW, 3e-5/3e-4 LR, batch 64, epochs 12-20
4. **Metrics**: ACC2_Non0 (primary), F1_Non0, MAE, Corr, ACC7
5. **Main results**: Tables from P6U freeze
6. **Baseline comparison**: Shows main model outperforms all baselines

## 5. Chapter 5 (Analysis) — Required Content

1. **Multimodal gain analysis**:
   - MOSI: +4.73% gain from audio → write as positive evidence
   - MOSEI: ~0% gain → write as text-dominant, cross-modal robustness
   - Discuss COVAREP limitations (74d, low-level acoustic, 90.3% alignment)

2. **AWAF weight analysis**:
   - MOSEI: gate=1.0 means audio weight near-zero
   - MOSI: gate values TBD (need MOSI main model AWAF data)

3. **Ablation experiments** (RECOMMENDED for P6U+):
   - text_only vs text_audio vs text_av (already done for MOSEI)
   - AWAF vs mean vs concat vs gated (MOSI needed)
   - sLSTM vs GRU vs no temporal encoder
   - LoRA r=16 vs r=8 vs full fine-tune

4. **Error analysis**:
   - Per-group accuracy (strong_neg, weak_neg, near_zero, weak_pos, strong_pos)
   - Confusion matrices

## 6. Excluded Items and Reasons

| Item | Reason |
|------|--------|
| MMIM-lite | Implementation identical to MISA-lite (byte-for-byte) |
| MOSEI TAV | Vision features all-zero (FACET unavailable) |
| P6S_repair4 8 baselines | COVAREP -inf collapse |
| P6N/P6O 8 MOSI baselines | Training collapse |
| P6S_repair2 main model | Empty outputs |
| COVAREP 74d as 768d | Factual error — it's 74d |

## 7. Writing Red Lines

1. Do NOT claim MOSEI audio improvement
2. Do NOT include MMIM-lite as separate model
3. Do NOT claim vision results on MOSEI
4. Do NOT claim baseline-lite is official reimplementation
5. Do NOT use smoke results in main table
6. Do NOT use collapse results
7. Do NOT write text_only as final main model
8. Do NOT confuse ACC2_Non0 with ACC2_Has0

## 8. Recommended Ablation Experiments (P6U+)

1. AWAF ablation: awaf vs mean vs concat vs gated vs fixed (MOSI)
2. sLSTM ablation: sLSTM vs GRU vs no temporal (MOSI+MOSEI)
3. Multi-seed: seed 2024 for main models
4. COVAREP vs alternative audio features (if available)

Generated: 2026-06-20
