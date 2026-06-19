# HANDOFF PHASE P6L

**Date**: 2026-06-19
**Branch**: `p6l-mosei-baseline-smoke`
**Base**: P6K `59b216f`

## Completed

1. ✅ MOSI mainline locked: text_audio conservative (T+A) — DECISIONS.md D039-D043
2. ✅ Dataset parameterization: training script supports `data.dataset`
3. ✅ MOSI backward compat smoke: passed
4. ✅ MLCL/DLF/DPDF-LQ repos cloned and audited
5. ✅ MMSA repo cloned and audited
6. ✅ MOSEI vision candidate plan written

## Blocked

1. ❌ MOSEI training: CSD feature extraction needed
2. ❌ Baseline smoke runs: need env setup
3. ❌ MOSEI dataloader: `data/mosei/label.csv` missing

## Next

1. Extract MOSEI features from CSD → label.csv + feature directories
2. Run MOSEI text_audio s42
3. Set up MLCL/MMSA envs, run smoke
4. Decision on MOSEI vision after text_audio results

## Key Decisions (see DECISIONS.md)

- D039: MOSI mainline = text_audio (T+A), vision exited
- D040: Residual/delta not primary gain source
- D041: 2025 baselines limited to MLCL/DLF/DPDF-LQ
- D042: MOSEI mainline not locked
- D043: Ch5 still blocked

## Files Changed

- `docs/DECISIONS.md` (+D039-D043)
- `scripts/train_textft_lora_mainline.py` (dataset parameterization)
- `reports/P6L_mosei_baseline_smoke/` (7 report files)
- `reports/experiment_registry.csv` (+7 entries)
