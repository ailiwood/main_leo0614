# P5D Checkpoint & Space Audit

> Date: 2026-06-17
> Total outputs size: 537MB
> Total .pth files: 34

## PTH Files by Run

| Run | best_model.pth | last_model.pth | Keep Best? | Keep Last? | Reason |
|-----|---------------|----------------|------------|------------|--------|
| P4T C0 s2024 | 12M | 12M | ✅ ref | ❌ delete | Old C0 baseline, keep best for reference |
| P4T C0 s42 | 12M | 12M | ✅ ref | ❌ delete | Old C0 baseline, keep best for reference |
| P4V c0_fixed seed2024 | 12M | 12M | ❌ delete | ❌ delete | Superseded by P4W |
| P4V c0_fixed seed42 | 12M | 12M | ❌ delete | ❌ delete | Superseded by P4W |
| P4V1 awaf_seq seed2024 | 16M | 16M | ❌ delete | ❌ delete | Exploration, superseded |
| P4V1 awaf_seq seed42 | 16M | 16M | ❌ delete | ❌ delete | Exploration, superseded |
| P4V1 bs32 seed42 | 16M | 16M | ❌ delete | ❌ delete | Batch size probe |
| P4V1 bs64 seed42 | 16M | 16M | ❌ delete | ❌ delete | Batch size probe |
| P4W A_baseline | 16M | 17M | ❌ delete | ❌ delete | Not best candidate |
| P4W A_baseline s2024 | 16M | 17M | ❌ delete | ❌ delete | Not best candidate |
| P4W B_larger_hidden | 33M | 33M | ❌ delete | ❌ delete | Not best candidate |
| P4W C_deeper_cross | 19M | 19M | ❌ delete | ❌ delete | Not best candidate |
| P4W D_larger_batch | 16M | 17M | ❌ delete | ❌ delete | Not best candidate |
| P4X ablation A_full | 16M | 17M | ❌ delete | ❌ delete | Exploration |
| P5A mosi_seed42 | 19M | 19M | ❌ delete | ❌ delete | Failed route (75.5%) |
| P5C smoke seed42 | 13M | 13M | ❌ delete | ❌ delete | 3ep smoke, not a real run |
| P5C seed42 | 13M | 13M | ✅ KEEP | ❌ delete | **Current best model** |

## Recommended Actions

### DELETE (30 files, ~480MB)
- All last_model.pth (17 files): not needed for reproduction
- All exploration/baseline best_model.pth from superseded runs (13 files)

### KEEP (4 files, ~48MB)
- P4T C0 best_model.pth ×2 (historical baseline reference)
- P5C seed42 best_model.pth (current best)
- P5D seed2024 best_model.pth (when training completes)

### PRESERVE (no action)
- All predictions_test.csv, awaf_weights_test.csv, text_base_delta_test.csv
- All test_metrics_final.json, val_metrics_epoch.csv, config.json
- All reports, logs, loss curves

## Space Savings
- Delete: ~480MB
- After cleanup: ~57MB (predictions + metrics + kept checkpoints)

## P5D Policy
- Default: save only best_model.pth (NO last_model.pth)
- Exploration runs: only keep CSVs, delete everything else
- Candidate runs: keep best_model.pth
- Never commit pth/pt files to Git
