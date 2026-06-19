# P6V-R: Runtime Process Audit

**Time**: 2026-06-20 01:40 UTC+8

## Active Python Processes: 3

## Currently Running Training

### 1. MOSI awaf_no_context — FORMAL TRAINING (PROTECT)
- **Status**: RUNNING, epoch 4/8
- **Output**: `outputs/P6V_ablation/mosi/awaf_no_context_s42_s42_20260620_013156`
- **Config**: `configs/experiments/p6v_ablation/mosi/awaf_no_context_s42.yaml`
- **Action**: ⏳ **WAIT** — Let complete. DO NOT KILL.
- **Note**: val_ACC2 stuck at 57.41% (all-positive). Likely collapsed. Let it finish for formal record.

## Completed Formal Training

### 2. MOSEI fusion_gated — COMPLETED
- **Status**: ✅ FORMAL COMPLETE, 4 epochs
- **ACC2**: 76.02%, F1: 83.08%, MAE: 0.746, Corr: 0.593
- **Delta vs main (87.86%)**: -11.84%
- **Output**: `outputs/P6V_ablation/mosei/fusion_gated_s42_s42_20260620_004227`
- **Missing**: predictions_test.csv (model architecture mismatch prevents re-eval with same config)
- **Action**: ✅ **KEEP** — Valid ablation result. Gated fusion significantly underperforms AWAF.

## Smoke-Only Outputs (EXCLUDE from formal table)

| Model | ACC2 | Epoch | Action |
|-------|------|-------|--------|
| MOSEI fusion_mean | 38.04% | 1 | ❌ Exclude — smoke only |
| MOSI fusion_mean | 57.77% | 1 | ❌ Exclude — smoke only |
| MOSI fusion_concat | 42.23% | 1 | ❌ Exclude — smoke only |
| MOSI fusion_fixed | 42.23% | 1 | ❌ Exclude — smoke only |
| MOSI fusion_gated | 42.23% | 1 | ❌ Exclude — smoke only |
| MOSI awaf_no_interaction | 42.23% | 1 | ❌ Exclude — smoke only |

## Pending / Not Started

| Dataset | Model | Action |
|---------|-------|--------|
| MOSEI | awaf_no_interaction | ⏳ Run formal |
| MOSEI | fusion_mean | ⏳ Run formal |
| MOSEI | encoder_gru | ⏳ Run formal |
| MOSEI | encoder_no_temporal | ⏳ Run formal |
| MOSI | fusion_mean | ⏳ Run formal |
| MOSI | fusion_concat | ⏳ Run formal |
| MOSI | fusion_gated | ⏳ Run formal |
| MOSI | awaf_no_interaction | ⏳ Run formal |
| MOSI | encoder_gru | ⏳ Run formal |
| MOSI | encoder_no_temporal | ⏳ Run formal |

## Inherited (no training needed)
| Dataset | Model | ACC2 | Source |
|---------|-------|------|--------|
| MOSI | main_awaf_slstm | 88.72% | P6K |
| MOSEI | main_awaf_slstm | 87.86% | P6T |

Generated: 2026-06-20
