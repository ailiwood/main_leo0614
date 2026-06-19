# P6U: Cleanup Plan

**Date**: 2026-06-20  
**Rule**: Move to `archive/P6U_cleanup_20260620/` NOT permanent delete.  
**Safety**: Keep manifest checked first. Smoke test after move.

## CANNOT DELETE (KEEP)

### Active Code
- `scripts/train_textft_lora_mainline.py` + `scripts/train_baseline_lite.py` + `scripts/eval_checkpoint.py`
- `models/textft_lora_xlstm_awaf_residual.py`
- `models/fusion/awaf.py` + `models/encoders/slstm.py` + `models/pooling/attention_pooling.py`
- `models/modules/minimal_lora.py` + `models/modules/uncertainty_residual_gate.py` + `models/modules/text_confidence_residual.py`
- `models/baselines/base_baseline.py` + 7 MOSEI baselines (NOT mmim_lite.py)
- `data/textft_multimodal_dataset.py` + `utils/metrics.py`
- `tests/test_metrics_mosei_non0.py`

### P6U Artifacts
- All `configs/experiments/p6u_freeze/` (14 YAML configs)
- All `reports/P6U_freeze/` (tables, figures, manifests, reports)
- All `scripts/p6u_*.py` (runner, status, compile)

### Best Model Outputs
- `outputs/P6S_repair5/mosei/baselines/` (7 MOSEI baselines)
- `outputs/P6T_prefreeze/mosei/mainline/text_audio_cached_s42_s42_20260619_222353/` (MOSEI main)
- `outputs/P6K/text_audio_conservative_s42_s42_20260619_031645/` (MOSI main)
- `outputs/P6P/baseline_rescue/mosi/` (MOSI baselines x2)
- `outputs/P6Q/baseline_rescue/mosi/` (MOSI baselines x2)
- `outputs/P6S_repair/mosei/text_only/` (MOSEI text_only diagnostic)

### Data
- `data/processed/mosei_full/` + `data/processed/mosei_full/roberta_cache/`
- `data/features_strong_sequence_mosi_v3_T40/`
- `data/mosi/label.csv`

### Git/Project
- All `.md` handoff files
- `CLAUDE.md`

## CAN MOVE TO ARCHIVE

### Collapse Outputs
- `outputs/_invalid/` (P6S_repair4 collapse)
- `outputs/P6N/` (MOSI collapse baselines)
- `outputs/P6O/` (MOSI collapse baselines)
- `outputs/P6S_repair3/` (intermediate collapse attempts)
- `outputs/P6S_repair4/mosei/baselines/` (now in _invalid, but check)

### Old Phase Outputs
- `outputs/P4*/` (early development)
- `outputs/P5*/` (early development)
- `outputs/P6C/`, `outputs/P6D/`, `outputs/P6E/`
- `outputs/P6H_repair/` (superseded by P6K)
- `outputs/P6I/` (superseded by P6K)
- `outputs/P6J/` (superseded)
- `outputs/P6S_repair2/` (empty outputs)

### Old Model Files (NOT in active dependency graph)
- `models/deeptext_xlstm_awaf_residual.py`
- `models/deeptext_xlstm_awaf_residual_v2.py`
- `models/textft_xlstm_awaf_residual.py`
- `models/baselines/mmim_lite.py`
- `models/ours_xlstm_fusion.py`

### Old Scripts
- `scripts/p6c_text_roberta_finetune.py`
- `scripts/p6d_mosei_text_roberta.py`
- `scripts/p6g_train_lora_mosi.py`
- `scripts/p6h_train_lora_main.py`
- `scripts/eval_deeptext_xlstm_awaf_residual.py`
- `scripts/diagnose_baseline_lite.py`
- `scripts/diagnose_mosei_collapse.py` (diagnostic tool — may keep for reference)

### Old Data Loaders
- `data/sequence_dataset.py`
- `data/strong_sequence_dataset.py`

### Old Reports (keep HANDOFFs, P6T, P6S_repair5, P6U)
- `review_packages/` (old code reviews)

### Empty/Stale
- Empty directories from old phases
- Duplicate config backups

## Verification After Cleanup
1. Run `python scripts/p6u_status.py`
2. Run import smoke tests
3. Run `python tests/test_metrics_mosei_non0.py`

Generated: 2026-06-20
