# HANDOFF_PHASE_04S.md

## Strict Protocol
✅ Fixed: val for model selection, test once at end. `engine/strict_trainer.py`

## Metrics
✅ Both cls_logit and reg_sign ACC2 now output. reg_sign = MMSA convention.

## Sequence Features
✅ MLCL standard MOSI features downloaded and audited.
- Text: word tokens (T~12), Audio: COVAREP 47d, Vision: FACET 74d

## CRITICAL FINDING
**MLCL sequence features (T~12) perform WORSE than TMDC T=1 features.**
- TMDC T=1: ACC2_NZ=75.25%, MAE=1.000, Corr=0.592
- MLCL T~12: ACC2_NZ=44.9%, MAE=1.493, Corr=-0.023

**Strong pretrained features (DeBERTa+wav2vec+CLIP 1024d) >> Weak aligned features (word_emb+COVAREP+FACET)**

## Root Cause
MLCL standard features use much weaker encoders:
- Text: 300d word embedding (vs DeBERTa-large 1024d)
- Audio: 47d COVAREP (vs wav2vec2-large 1024d)
- Vision: 74d FACET (vs CLIP-ViT 1024d)

## Recommendation
Use TMDC-v1 1024d features for paper. T=1 limitation should be discussed but does not invalidate the model.
For true sequence validation, use BERT token-level or wav2vec frame-level features (not GloVe/COVAREP).

## Next Steps
1. Accept TMDC-v1 as primary features
2. Proceed to P5 (ablation) with TMDC features
3. Optional: explore BERT sequence features as supplement
