# P6V: P6U Fact Check for Ablation

**Date**: 2026-06-20  
**Branch**: `p6v-core-ablation-fast`

## Confirmed P6U Facts

| # | Fact | Status |
|---|------|--------|
| 1 | MOSI Main text_audio ACC2=88.72% | ✅ P6K |
| 2 | MOSEI Main text_audio ACC2=87.86% | ✅ P6T |
| 3 | MOSEI text_only is diagnostic only | ✅ Not final model |
| 4 | MOSEI TAV excluded (vision all-zero) | ✅ |
| 5 | MMIM-lite excluded (MISA duplicate) | ✅ |
| 6 | P6V ablation only — no P6U modification | ✅ |
| 7 | P6V output to independent dirs | ✅ outputs/P6V_ablation/ |

## Ablation Matrix

### MOSI (10 configs, 8 epoch)
1. main_awaf_slstm (inherited from P6K)
2. fusion_mean (smoke passed ✅, training 🔄)
3. fusion_concat (training 🔄)
4. fusion_gated (training 🔄)
5. fusion_fixed (training 🔄)
6. awaf_no_interaction (training 🔄)
7. awaf_no_context (training 🔄)
8. encoder_gru (training 🔄)
9. encoder_lstm (training 🔄)
10. encoder_no_temporal (training 🔄)

### MOSEI (6 configs, 4 epoch)
1. main_awaf_slstm (inherited from P6T)
2. fusion_mean (smoke passed ✅, training 🔄)
3. fusion_gated (training 🔄)
4. awaf_no_interaction (training 🔄)
5. encoder_gru (training 🔄)
6. encoder_no_temporal (training 🔄)

## Data Fix Applied
- `data/textft_multimodal_dataset.py`: Fixed `split_map` to try both 'valid' and 'val' directories for feature files
- Root cause: MOSI features use 'val' dir, MOSEI uses 'valid' dir
- Fix: tries multiple candidate directory names

## Training Status
- MOSEI: 🔄 5 models running (background PID: bn9duro8n)
- MOSI: 🔄 8 models running (background PID: b0o4bibbc)
- Estimated total time: ~3 hours

Generated: 2026-06-20
