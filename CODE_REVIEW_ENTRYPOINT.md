# CODE_REVIEW_ENTRYPOINT.md

## For Web AI: Priority Reading Order

1. **README_PROJECT_STATUS.md** — current project state
2. **CODE_REVIEW_ENTRYPOINT.md** (this file) — quick overview
3. **memory.md** — project memory & phase history
4. **docs/DECISIONS.md** — key decisions D001-D0xx
5. **经验总结.md** — lessons learned
6. **HANDOFF_PHASE_04T1.md** — latest handoff (strong seq + C0 retrain)
7. **models/ours_xlstm_fusion.py** — main model (sLSTM + AWAF)
8. **models/encoders/slstm.py** — sLSTM implementation
9. **models/fusion/awaf.py** — AWAF fusion module
10. **engine/strict_trainer.py** — strict protocol trainer
11. **data/strong_sequence_dataset.py** — strong sequence dataset
12. **scripts/extract_mosi_strong_sequence.py** — feature extraction
13. **utils/metrics.py** — unified metrics
14. **configs/models/ours_c0_best.yaml** — C0 config

## Core Problems to Solve

1. **C0 on strong sequence is still low (~72.7% mean ACC2_NZ_reg)**
   - Text: DeBERTa-large token-level (T~18, 1024d)
   - Audio: wav2vec2-base frame-level (T~93, 768d)
   - Vision: CLIP pre-extracted .pt (T=1, 1024d)

2. **Vision is still T=1** — need frame-level sequence features
3. **Model lacks sequence-level cross-modal interaction**
   - AWAF operates on pooled utterance representations
   - No cross-modal attention at sequence level
4. **Need AWAF-Seq + Cross-modal Transformer design**

## Questions for Web AI

1. Is the current C0 architecture correct?
2. Is sLSTM implementation suitable for this task?
3. Does AWAF match the paper's definition?
4. Is strict protocol reliable?
5. Should we upgrade wav2vec2-base → wav2vec2-large?
6. How to get vision T>1 features?
7. How to design AWAF-Seq + Cross-modal Transformer?
8. Path to 85+ ACC2_NZ on MOSI?
