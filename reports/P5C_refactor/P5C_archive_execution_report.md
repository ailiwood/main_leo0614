# P5C Archive Execution Report

> Phase: P5C — DeepText-xLSTM-AWAF Residual Refactor
> Archive: archives/P5C_pre_refactor_exploration_06171310/
> Date: 2026-06-17

## Summary

| Category | Action | Count |
|----------|--------|-------|
| Old models | MOVED | 6 files (3 main + 3 enhancements) |
| Old scripts | MOVED | 10 files |
| Old configs | MOVED | 10 files |
| Old reports | COPIED | 20+ dirs/files |
| Old handoffs | COPIED | 16 files |
| Core modules | PRESERVED | 11 files |

## Actions

### MOVED (removed from active code tree)
- models/ours_xlstm_fusion.py → archive/models_exploration/
- models/ours_awaf_seq_xlstm.py → archive/models_exploration/
- models/ours_text_guided_awaf_xlstm.py → archive/models_exploration/
- models/enhancements/* (cme.py, cme_residual.py, deconv.py) → archive/models_exploration/enhancements/
- models/interaction/modality_reliability.py → archive/models_exploration/interaction/
- scripts/p3a_hparam_search.py, p3b_candidate_runs.py → archive/scripts_exploration/
- scripts/train_main.py, train_awaf_seq.py, train_text_guided_awaf_xlstm.py → archive/scripts_exploration/
- scripts/smoke_forward.py, test_awaf_seq_modules.py, test_text_guided_awaf_xlstm.py → archive/scripts_exploration/
- scripts/extract_mosi_data2vec_audio.py, extract_mosi_strong_sequence.py → archive/scripts_exploration/
- configs/models/ours_*.yaml (8 files) → archive/configs_exploration/
- configs/ours_text_guided_awaf_xlstm_mosi.yaml, ours_text_guided_awaf_xlstm_mosei_sdk.yaml → archive/configs_exploration/

### COPIED (originals preserved for evidence chain)
- HANDOFF_PHASE_00.md through HANDOFF_PHASE_10B.md → archive/handoffs_exploration/
- reports/P0_*.md through reports/P5B_*/ → archive/reports_exploration/

### PRESERVED (untouched)
- models/encoders/slstm.py, models/fusion/awaf.py, models/heads.py
- models/interaction/cross_modal_transformer.py, models/pooling/attention_pooling.py
- engine/strict_trainer.py
- utils/metrics.py, utils/seed.py
- data/strong_sequence_dataset.py, data/dataset.py, data/sequence_dataset.py
- configs/default.yaml, configs/data/mosi_strong_sequence.yaml

## File Manifest
- archives/P5C_pre_refactor_exploration_06171310/file_manifest_before_move.csv (99 entries)

## Verification
- models/ directory contains only active modules
- No __pycache__ referencing deleted files (will be cleaned on next import)
- All moved files accessible in archive
