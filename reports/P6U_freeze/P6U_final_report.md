# P6U: Strict Freeze + Cleanup — Final Report

**Date**: 2026-06-20  
**Branch**: `p6u-strict-freeze-cleanup`  
**Status**: ✅ COMPLETE

---

## 1. Final Results — Freeze-Ready (14 Models)

### MOSEI (9 models)
| Model | Type | ACC2_Non0 | F1_Non0 | MAE | Corr |
|-------|------|-----------|---------|-----|------|
| **Main (ours)** | main | **87.86%** | **90.25%** | **0.519** | **0.800** |
| Main (ours) | diagnostic | 88.22% | 90.57% | 0.509 | 0.813 |
| MISA-lite | baseline | 82.67% | 86.62% | 0.621 | 0.687 |
| SelfMM-lite | baseline | 82.60% | 86.33% | 0.623 | 0.682 |
| MulT-lite | baseline | 82.39% | 86.17% | 0.611 | 0.705 |
| LMF-lite | baseline | 81.85% | 85.67% | 0.624 | 0.682 |
| TFN-lite | baseline | 81.48% | 85.69% | 0.638 | 0.664 |
| MLCL-lite | baseline | 80.66% | 84.24% | 0.656 | 0.649 |
| DLF-lite | baseline | 77.90% | 82.98% | 0.733 | 0.555 |

### MOSI (5 models)
| Model | Type | ACC2_Non0 | F1_Non0 | MAE | Corr |
|-------|------|-----------|---------|-----|------|
| **Main (ours)** | main | **88.72%** | **86.64%** | **0.635** | **0.851** |
| TFN-lite | baseline | 77.59% | 71.01% | 1.078 | 0.614 |
| MulT-lite | baseline | 77.13% | 71.80% | 0.942 | 0.641 |
| SelfMM-lite | baseline | 76.68% | 73.11% | 1.024 | 0.614 |
| LMF-lite | baseline | 75.61% | 72.41% | 1.045 | 0.586 |

## 2. Excluded Models
- MMIM-lite (byte-identical to MISA-lite)
- MOSEI TAV (vision all-zero)
- P6S_repair4 8 baselines (COVAREP -inf collapse)
- P6N/P6O 8 baselines (training collapse)

## 3. Generated Artifacts

### Tables
- `reports/P6U_freeze/tables/main_results_all.csv`
- `reports/P6U_freeze/tables/mosi_results.md`
- `reports/P6U_freeze/tables/mosei_results.md`

### Figures (1000 DPI)
- `reports/P6U_freeze/figures/mosei_main_vs_baseline.png`
- `reports/P6U_freeze/figures/mosi_main_vs_baseline.png`
- `reports/P6U_freeze/figures/mosei_text_only_vs_text_audio.png`
- `reports/P6U_freeze/figures/baseline_acc2_comparison.png`

### Configs (14 YAML)
- `configs/experiments/p6u_freeze/` — all freeze configs

### Scripts
- `scripts/p6u_run_all_smoke.py` — smoke test runner
- `scripts/p6u_status.py` — status checker
- `scripts/p6u_compile_results.py` — results compiler
- `scripts/p6u_generate_freeze_configs.py` — config generator

### Manifests
- `reports/P6U_freeze/manifests/active_code_manifest.md`
- `reports/P6U_freeze/manifests/cleanup_plan.md`

## 4. Cleanup Summary

**Moved to archive**: 40+ old phase outputs, 5 old model files, 6 old scripts, 2 old data loaders.
- Path: `archive/P6U_cleanup_20260620/`
- Restored: P6P, P6Q, P6R (MOSI baselines), P6S_repair (MOSEI text_only)

**Post-cleanup verification**: ✅ All 11 import tests pass, 6/6 critical artifacts exist, 8/8 unit tests pass.

## 5. Git Status

```
Branch: p6u-strict-freeze-cleanup
Base: p6s-repair4-mosei-fast-finish
```

### To commit:
- All P6U reports, configs, scripts, figures, manifests
- Modified source files (collate fix, metrics fix, collapse guard, smoke flags)

### NOT committed (large files):
- `data/processed/mosei_full/roberta_cache/*.npy` (~85MB)
- `outputs/**/best_model.pth` (large weights)
- `outputs/**/last_model.pth`

## 6. Recovery Commands

```bash
# Restore from archive if needed
cp -r archive/P6U_cleanup_20260620/<name> ./

# Run smoke test
python scripts/p6u_run_all_smoke.py --device cuda

# Run status check
python scripts/p6u_status.py

# Unit tests
python tests/test_metrics_mosei_non0.py
```

## 7. Next Steps (P6V — Ablation + Paper Writing)
1. Ablation experiments (AWAF variants, sLSTM vs GRU, multi-seed)
2. Generate thesis tables from P6U freeze results
3. Write Chapter 3 (method) from code
4. Write Chapter 4 (experiments) from P6U results
5. Write Chapter 5 (analysis) with error analysis and weight interpretation

Generated: 2026-06-20
