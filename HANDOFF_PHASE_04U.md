# HANDOFF_PHASE_04U.md — P4U Cleanup & GitHub Upload

## Deleted Files Summary

### Old Code (~15 files)
- V9 route: model.py, train.py, dataset.py, balanced_sampler.py, ensemble_5seed.py, extract_features.py
- Old utils: utils_models/ (lstm_v.py AGPL, ours_model.py, DEConv.py, etc.), utils_tools/, utils_train/
- Adapter: tmdc_adapter/ (all extraction scripts, hf_cache, features)

### Old Results & Outputs (~30 dirs)
- results_20260307/ (first-round results, images, JSON)
- experiments/v9_roberta*/ (V9 weights ~115MB, deleted locally)
- outputs/P2_smoke, P3A*, P3B*, P3C*, P4A*, P4S*
- logs/ (old training logs)

### Cache & Temp
- Word temp file, old reference PDFs, old startup prompts

## Preserved (Core)

### Code (12 files)
models/ (encoders/slstm, fusion/awaf, heads, ours_xlstm_fusion, enhancements/)
engine/ (strict_trainer, trainer)
data/ (dataset, sequence_dataset, strong_sequence_dataset)
utils/ (metrics, seed)

### Configs & Scripts
configs/ (5 yaml), scripts/ (6 py, including extract_mosi_strong_sequence)

### Research Records
HANDOFF_PHASE_*.md (00-04U), memory.md, docs/DECISIONS.md, docs/经验总结.md,
CODE_REVIEW_ENTRYPOINT.md, README_PROJECT_STATUS.md, reports/ (key summaries)

## GitHub Upload

| Item | Value |
|------|-------|
| Remote | https://github.com/ailiwood/main_leo0614.git |
| Branch | **p4u-clean-before-architecture-upgrade** |
| Commit | **08c67fa** |
| Status | ✅ Pushed |
| Files | 96 new, 214 total changes |

## Post-Delete Integrity
All 6 import tests passed, scripts exist, configs exist, random forward/backward OK.

## Next Steps
Web AI reviews GitHub repo → designs AWAF-Seq + Cross-modal Transformer upgrade.
Entry point: CODE_REVIEW_ENTRYPOINT.md
