# P4R Architecture Upgrade Plan

## Root Cause of Low Performance (75.25% vs MMSA ~85%)

**Primary bottleneck: T=1 clip-level features.**
sLSTM with T=1 is functionally equivalent to a feed-forward layer — it cannot model temporal dependencies.
AWAF operates on pooled single vectors, not sequence-level cross-modal interaction.
This fundamentally limits performance regardless of model design.

## Three Routes

### Route A: Minimum Fix (Effort: Low, Expected Gain: +2-4%)

1. Fix test leakage → use val for model selection
2. Switch ACC2 to reg_pred sign (MMSA convention)
3. Add sign consistency loss between reg and cls
4. Small LR sweep: 5e-5, 1e-4, 2e-4
5. **No architecture change**

Expected MOSI ACC2_Non0: ~78-80%

### Route B: Feature Upgrade (Effort: Medium, Expected Gain: +5-10%) [RECOMMENDED]

1. Use MMSA/CMU-MultimodalSDK standard sequence features
   - Text: GloVe 300d or BERT 768d, aligned word-level
   - Audio: COVAREP 74d or wav2vec 1024d, frame-level
   - Vision: FACET 35d or OpenFace, frame-level
   - T > 1, typically 50-64 frames
2. sLSTM can now model real temporal dynamics
3. AWAF operates on sequence-pooled vectors (still valid)
4. Add optional cross-modal attention between sequences
5. **Minimal model change — feature pipeline change only**

Required files: data/feature loader, possibly download script from MLCL/MMSA
Expected MOSI ACC2_Non0: ~82-86%

### Route C: Model Upgrade (Effort: High, Expected Gain: +8-12%)

1. All of Route B
2. Add cross-modal transformer at sequence level (before AWAF)
3. Add unimodal auxiliary supervision (Self-MM style)
4. Add contrastive loss between modalities (MLCL style)
5. Deeper sLSTM or bidirectional

Risk: May overfit on small MOSI dataset. Requires significant code changes.

## Recommendation

**Route B first.** Get sequence features working → validate sLSTM temporal benefit → then optionally add small model upgrades.
