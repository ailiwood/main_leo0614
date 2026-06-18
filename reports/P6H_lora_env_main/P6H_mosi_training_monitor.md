# P6H MOSI MinimalLoRA-AWAF Training Monitor — FINAL

**Last update**: 2026-06-18 17:04 UTC+8  
**Status**: ✅ **COMPLETED**

---

## Training Timeline

| Event | Time |
|---|---|
| Started | 2026-06-18 14:19 |
| Completed | 2026-06-18 15:23 |
| Duration | ~64 min |
| Output | `outputs/P6H/mosi/lora_seed42.json` |

---

## Final Summary

| Metric | Value |
|---|---|
| text_base_ACC2_Non0 | 86.43% |
| final_ACC2_Non0 | 86.43% |
| residual_gain | **0.00%** |
| F1_Non0 | 84.36% |
| MAE | 0.6775 |
| Corr | 0.8275 |
| ACC7 | 45.92% |
| best_val_ACC2 | 87.50% |

---

## Critical Findings

1. **AWAF weights collapsed**: w_v=0.984, w_t=0.006, w_a=0.010 → vision-dominant
2. **Delta scale too small**: dsr=0.02 → residual magnitude ~0.005 (negligible)
3. **Gate suppression**: gate_mean=0.46 → half of residual blocked
4. **Residual has zero effect**: final == text_base exactly

---

## Decision

**介于 Case C 与 D 之间** — text_base 健康 (86.43%)，但多模态残差融合完全无效。

→ 进入调参阶段，P0 修复：AWAF LayerNorm + τ=3.0 + dsr=0.2  
→ 不启动 MOSEI，不补多 seed，不启动 baseline
