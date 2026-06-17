# P5C Archive Plan — Old Exploration File Archiving

> Phase: P5C DeepText-xLSTM-AWAF Residual Refactor
> Target: archives/P5C_pre_refactor_exploration_06171310/
> Principle: Copy reports/handoffs (preserve evidence chain), Move old code (prevent confusion)

## Core Modules PRESERVED (stay in place)

| Path | Reason |
|------|--------|
| models/encoders/slstm.py | Core: sLSTM encoder (used in audio/vision branches) |
| models/fusion/awaf.py | Core: AWAF fusion (used as residual weight generator) |
| models/heads.py | Core: prediction heads (may need residual variants) |
| models/interaction/cross_modal_transformer.py | Core: cross-modal transformer (optional module) |
| models/pooling/attention_pooling.py | Core: masked attention pooling |
| engine/strict_trainer.py | Core: strict protocol trainer |
| utils/metrics.py | Core: unified metrics |
| utils/seed.py | Core: random seed control |
| data/strong_sequence_dataset.py | Core: strong sequence data loading |
| data/dataset.py | Core: T=1 dataset |
| configs/default.yaml | Core: default config |
| configs/data/mosi_strong_sequence.yaml | Core: data config |

## Files to MOVE to archive

### Old Model Implementations (superseded by deeptext_xlstm_awaf_residual.py)
| Source | Archive Dest | Reason |
|--------|-------------|--------|
| models/ours_xlstm_fusion.py | models_exploration/ | P2-P4 three-way sLSTM fusion (superseded) |
| models/ours_awaf_seq_xlstm.py | models_exploration/ | P4V-P4W AWAF-Seq (superseded) |
| models/ours_text_guided_awaf_xlstm.py | models_exploration/ | P5A Text-Guided (failed) |
| models/enhancements/cme.py | models_exploration/enhancements/ | Candidate module (not adopted) |
| models/enhancements/cme_residual.py | models_exploration/enhancements/ | Candidate module (not adopted) |
| models/enhancements/deconv.py | models_exploration/enhancements/ | Candidate module (not adopted) |
| models/enhancements/__init__.py | models_exploration/enhancements/ | Part of enhancements |
| models/interaction/modality_reliability.py | models_exploration/interaction/ | P5A exploration |

### Old Scripts (superseded by new P5C scripts)
| Source | Archive Dest | Reason |
|--------|-------------|--------|
| scripts/p3a_hparam_search.py | scripts_exploration/ | P3 exploration |
| scripts/p3b_candidate_runs.py | scripts_exploration/ | P3 exploration |
| scripts/train_main.py | scripts_exploration/ | Old training entry |
| scripts/train_awaf_seq.py | scripts_exploration/ | Old P4V training |
| scripts/train_text_guided_awaf_xlstm.py | scripts_exploration/ | Old P5A training |
| scripts/smoke_forward.py | scripts_exploration/ | Old P2 smoke test |
| scripts/test_awaf_seq_modules.py | scripts_exploration/ | Old P4V unit test |
| scripts/test_text_guided_awaf_xlstm.py | scripts_exploration/ | Old P5A unit test |
| scripts/extract_mosi_data2vec_audio.py | scripts_exploration/ | Old extraction script |
| scripts/extract_mosi_strong_sequence.py | scripts_exploration/ | Old extraction script |

### Old Configs (superseded)
| Source | Archive Dest | Reason |
|--------|-------------|--------|
| configs/models/ours_awaf_seq_xlstm.yaml | configs_exploration/ | Old P4V config |
| configs/models/ours_backbone.yaml | configs_exploration/ | Old P2 config |
| configs/models/ours_c0_best.yaml | configs_exploration/ | Old P3 config |
| configs/models/ours_c0_p4a_final.yaml | configs_exploration/ | Old P4A config |
| configs/models/ours_c0_strong_sequence.yaml | configs_exploration/ | Old P4T config |
| configs/models/ours_c1_deconv.yaml | configs_exploration/ | Old P3B config |
| configs/models/ours_c3_cme.yaml | configs_exploration/ | Old P3B config |
| configs/models/ours_text_guided_awaf_xlstm_mosi.yaml | configs_exploration/ | Old P5A config |
| configs/ours_text_guided_awaf_xlstm_mosi.yaml | configs_exploration/ | Old P5A root config |
| configs/ours_text_guided_awaf_xlstm_mosei_sdk.yaml | configs_exploration/ | Old P5A MOSEI config |

### Old Reports & Handoffs (COPY — keep originals for evidence chain)
| Source | Archive Dest | Reason |
|--------|-------------|--------|
| HANDOFF_PHASE_00.md through HANDOFF_PHASE_10B.md | handoffs_exploration/ | P0-P5B handoffs (historical record) |
| reports/P0_*.md through reports/P4Z_*/ | reports_exploration/ | P0-P4Z reports (historical record) |
| reports/P3A_*/ reports/P3B_*/ reports/P3C_*/ | reports_exploration/ | P3 exploration reports |
| reports/P4A_*/ reports/P4R_*/ ... reports/P4Z_*/ | reports_exploration/ | P4 exploration reports |
| reports/P5A_*/ reports/P5B_*/ | reports_exploration/ | P5A/P5B reports |

## Files NOT to archive or delete
- data/features_strong_sequence_mosi*/ (active feature data)
- data/mosi/ (raw data)
- data/raw/ (raw videos)
- data/CMU-MOSEI/ (MOSEI data)
- outputs/ (experiment outputs)
- env/ (environment configs)
- .git/ (git repo)
- data/*.py (active data modules)
