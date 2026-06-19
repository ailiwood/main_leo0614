# HANDOFF PHASE P6M

**Date**: 2026-06-19
**Branch**: `p6m-baseline-lite-policy`
**Base**: P6L `252738c`

## Completed

1. ✅ D044-D051 written to DECISIONS.md
2. ✅ memory.md updated (0.5 Baseline-Lite route)
3. ✅ models/baselines/ created (__init__.py + base_baseline.py)
4. ✅ baseline_table_policy.md — 3-tier result classification
5. ✅ EXTERNAL_BASELINE_REFERENCES.md
6. ✅ P6M_config_update_report.md
7. ✅ Directory structure for configs/experiments

## Key Policy Changes

- Baseline route: external repos → internal baseline-lite
- Paper tables: 3 tiers (Primary Model / Baseline-Lite / Reported Original)
- Fixed baselines: 8 models (TFN/LMF/MulT/MISA/SelfMM/MMIM/MLCL/DLF-lite)
- CASP excluded from main table
- DPDF-LQ/DashFusion/R3DG deferred

## Not Yet Done

- Individual baseline-lite model files (tfn_lite.py etc.)
- Baseline-lite training script
- Per-model YAML configs
- Actual baseline training

## Next Phase

Implement baseline-lite models in order: TFN → LMF → MulT → SelfMM
