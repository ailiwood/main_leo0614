# P6H-R Full seed42 Best Config Result — R2 (entropy regularization)

**Date**: 2026-06-18  
**Config**: R2 entropy (τ=3.0, dsr=0.2, gate_bias=2.0, λ_entropy=0.01)  
**Epochs**: 42 (early stop at E42, best at E32)

---

## 1. Core Test Metrics

| Metric | P6H Original | P6H-R Best | Δ |
|---|---|---|---|
| text_base_ACC2_Non0 | **86.43%** | **81.86%** | **−4.57%** 🔴 |
| final_ACC2_Non0 | 86.43% | 81.86% | −4.57% |
| **residual_gain** | 0.00% | **0.00%** | 0.00% |
| F1_Non0 | 84.36% | 80.00% | −4.36% |
| MAE | 0.678 | 0.957 | +0.279 🔴 |
| Corr | 0.828 | 0.729 | −0.099 |
| ACC7 | 45.92% | 35.13% | −10.79% |

---

## 2. AWAF Weights (Repair: ✅ SUCCESS)

| Metric | P6H Original | P6H-R Best |
|---|---|---|
| w_t | 0.006 (0.6%) 🔴 | **0.263 (26.3%)** ✅ |
| w_a | 0.010 (1.0%) 🔴 | **0.228 (22.8%)** ✅ |
| w_v | 0.984 (98.4%) 🔴 | **0.510 (51.0%)** 🟡 |
| AWAF entropy | ~0.0 | **0.993** (max=1.099) ✅ |
| τ (temperature) | 1.0 | **3.08** ✅ |

> **AWAF 权重修复成功！** 不再单模态崩溃，三模态权重分配合理。

---

## 3. Gate & Delta (Repair: ❌ FAILED)

| Metric | P6H Original | P6H-R Best |
|---|---|---|
| gate_mean | 0.460 | 0.509 |
| delta_scale_reg | 0.020 (fixed) | 0.250 (learned) |
| effective_delta_abs_mean | 0.005 | 0.021 |
| effective_residual ≈ gate×dsr×|\|δ\|| | 0.005 | 0.006 |

> **残差贡献仍可忽略。** Gate 抑制 50%，有效 delta 仅 0.02。

---

## 4. Group Error Analysis

| Group | N | tb_acc | fn_acc | Δ |
|---|---|---|---|---|
| strong_neg | 202 | 87.13 | 85.64 | −1.5% |
| weak_neg | 165 | 72.73 | 72.73 | 0 |
| near_zero | 64 | 53.12 | 54.69 | +1.6% |
| weak_pos | 135 | 80.74 | 80.00 | −0.7% |
| strong_pos | 120 | 97.50 | 97.50 | 0 |

> Residual 对 near_zero 略有帮助 (+1.6%), 但对 strong_neg 有损 (−1.5%).

---

## 5. Val/Test Gap

| Set | ACC2 |
|---|---|
| best_val | **88.89%** |
| test | **81.86%** |
| **gap** | **+7.03%** 🔴 |

> 严重的 val/test 不匹配。Val set (229 samples) 过小，低 LoRA lr 可能导致过拟合。

---

## 6. 与 P6H Original 对比

| 维度 | P6H | P6H-R | 判断 |
|---|---|---|---|
| text_base ACC2 | 86.43% | 81.86% | 🔴 退化 4.57% |
| residual_gain | 0.00% | 0.00% | ❌ 未修复 |
| AWAF collapse | YES (w_v=98%) | NO | ✅ 已修复 |
| LoRA lr | 5e-5 | 3e-6 | 🔴 1/17 导致 text 欠训 |
| val_ACC2 | 87.50% | 88.89% | 🟡 微升但不可信 |

---

## 7. 分流判断: **Case D — 修复失败**

- final_ACC2 (81.86%) = text_base_ACC2 (81.86%) → residual_gain = 0.00%
- text_base 从 86.43% 退化到 81.86%（低 LoRA lr 所致）
- 不启动 MOSEI
- 不补 seed
- 需结构级救援

---

## 8. 已生成产物

| 文件 | 状态 |
|---|---|
| result.json | ✅ |
| metrics_epoch.csv (42 epochs) | ✅ |
| predictions_test.csv | ✅ |
| awaf_weights_test.csv | ✅ |
| text_base_delta_test.csv | ✅ |
| group_error_analysis.csv | ✅ |
| gate_delta_stats.csv | ✅ |
| mosi_training_curves.png | ✅ |
| mosi_awaf_weight_distribution.png | ✅ |
| mosi_gate_distribution.png | ✅ |
| mosi_delta_distribution.png | ✅ |
| mosi_text_vs_final_scatter.png | ✅ |
| mosi_confusion_matrix.png | ✅ |
