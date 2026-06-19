# P6V-R: Remaining Ablation Plan

**Time**: 2026-06-20 01:45 UTC+8  
**Currently Running**: MOSI awaf_no_context (epoch 4/8, ~20 min remaining)

## After Current Training Completes

### Priority 1: MOSI Core (7 models including inherited)
| # | Model | Status | Action |
|---|-------|--------|--------|
| 1 | main_awaf_slstm | Inherited (88.72%) | ✅ Done |
| 2 | awaf_no_interaction | Smoke only | 🔄 Run formal (8ep) |
| 3 | fusion_mean | Smoke only | 🔄 Run formal (8ep) |
| 4 | fusion_concat | Smoke only | 🔄 Run formal (8ep) |
| 5 | fusion_gated | Smoke only | 🔄 Run formal (8ep) |
| 6 | encoder_gru | Not started | 🔄 Run formal (8ep) |
| 7 | encoder_no_temporal | Not started | 🔄 Run formal (8ep) |
| 8* | awaf_no_context | Running (ep 4/8) | ⏳ Wait |

*awaf_no_context may collapse (val stuck at 57.41%). If collapse confirmed, record as failed_collapse, do not re-run.

### Priority 2: MOSEI Compact (6 models)
| # | Model | Status | Action |
|---|-------|--------|--------|
| 1 | main_awaf_slstm | Inherited (87.86%) | ✅ Done |
| 2 | fusion_gated | Formal (76.02%) | ✅ Done |
| 3 | fusion_mean | Smoke only | 🔄 Run formal (4ep) |
| 4 | awaf_no_interaction | Not started | 🔄 Run formal (4ep) |
| 5 | encoder_gru | Not started | 🔄 Run formal (4ep) |
| 6 | encoder_no_temporal | Not started | 🔄 Run formal (4ep) |

### Priority 3: Optional (time permitting)
- MOSI fusion_fixed
- MOSI encoder_lstm

## Launch Order (sequential, single GPU)
```
1. MOSI awaf_no_interaction (8ep)
2. MOSI fusion_mean (8ep)
3. MOSI fusion_concat (8ep)
4. MOSI fusion_gated (8ep)
5. MOSI encoder_gru (8ep)
6. MOSI encoder_no_temporal (8ep)
7. MOSEI fusion_mean (4ep)
8. MOSEI awaf_no_interaction (4ep)
9. MOSEI encoder_gru (4ep)
10. MOSEI encoder_no_temporal (4ep)
```

Estimated total: ~2.5 hours for all 10 remaining experiments.

## Time Estimates Per Model
- MOSI (8 epochs): ~4-5 min each
- MOSEI (4 epochs): ~25-30 min each

Generated: 2026-06-20
