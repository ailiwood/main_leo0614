# P6H MOSI MinimalLoRA-AWAF seed42 完整结果

**Generated**: 2026-06-18 17:04 UTC+8  
**Training completed**: 2026-06-18 15:23 UTC+8  
**Training duration**: ~64 min (14:19 → 15:23)

---

## 1. 核心指标

| 指标 | 值 | 说明 |
|---|---|---|
| **text_base_ACC2_Non0** | **86.43%** | RoBERTa+LoRA text-only 基线 |
| **final_ACC2_Non0** | **86.43%** | 三模态 AWAF+Gate+Delta Residual |
| **residual_gain** | **0.00%** | ⚠️ 残差融合无效果 |
| F1_Non0 | 84.36% | |
| ACC2_Has0 | — | 脚本未输出 (需补) |
| F1_Has0 | — | 脚本未输出 |
| MAE | 0.6775 | |
| Corr | 0.8275 | |
| ACC7 | 45.92% | |
| best_val_ACC2 | 87.50% | 验证集最优 epoch |

---

## 2. AWAF 权重分析

| 模态 | 平均权重 | 状态 |
|---|---|---|
| Text (w_t) | **0.0056** | 🔴 几乎为零 |
| Audio (w_a) | **0.0100** | 🔴 几乎为零 |
| Vision (w_v) | **0.9844** | 🔴 几乎独占 |

> **关键诊断**：AWAF 权重已崩溃为 vision-only 模式。文本仅有 0.56% 权重，音频 1.0%，视觉 98.4%。这说明 AWAF 在训练中学会几乎完全忽略文本和音频，但 visual features 是 frozen 768d → Linear → sLSTM 的轻量管线，表现力不足以产生有效残差。

---

## 3. Gate & Delta 分析

| 参数 | 值 | 诊断 |
|---|---|---|
| gate_mean (regression) | 0.4596 | 中等抑制 (~46% 通过) |
| delta_scale_init (dsr/dsc) | 0.02 | 🔴 过小 |
| max_delta | 0.5 | 合理 |
| 有效残差上限 | 0.46 × 0.02 × 0.5 = **0.0046** | 🔴 几乎可忽略 |

> 即使 gate=1.0，有效残差上限也仅 0.01（因为 dsr=0.02），不足以在回归值上产生 ~0.5-1.0 量级的变化。

---

## 4. 训练配置

```text
DEVICE   = cuda
BATCH    = 4
ACCUM    = 4 (eff ~16)
EPOCHS   = 50
LR       = 5e-5
SEED     = 42
H        = 256
LoRA r   = 16, α = 32
Trainable = 4.64M / 357M total
```

---

## 5. 结果判断

| 判断 | 结论 |
|---|---|
| text_base 是否健康 (>85%)？ | ✅ 是 (86.43%) |
| final 是否超过 text_base？ | ❌ 否 (完全相同) |
| 残差融合是否有效？ | ❌ 否 (gain=0.00%) |
| AWAF 权重是否合理？ | ❌ 否 (vision 崩溃) |
| 是否超过 87%？ | ❌ 否 |
| 是否超过 88%？ | ❌ 否 |

---

## 6. 根因分析

三条并发根因导致残差融合完全无效：

### 根因 1：AWAF 权重崩溃
- w_v=0.984，w_t=0.006，w_a=0.010
- 原因：vision features 经 sLSTM 后可能含有更大方差/范数，softmax 被视觉特征主导
- 需引入：entropy regularization、temperature τ 调节、或权重范围约束

### 根因 2：Delta scale 过小
- dsr=0.02 → 即使 gate 全开残差也几乎为零
- 建议：dsr_init=0.1~0.2，或让 delta_scale 可学习且不初始化为极小值

### 根因 3：Gate 抑制
- gate_mean=0.46 → 大约一半残差被门控抑制
- 如果 delta 修好后 gate 仍低，需引入 gate_bias 推动初始 gate 更接近 1.0

---

## 7. 分流

按决策规则，final_ACC2 == text_base_ACC2 = 86.43% < 87%。

**实际分类：介于 C 与 D 之间** — 残差融合未产生任何增益（非破坏），AWAF 权重已崩溃。

输出文件：
- [x] P6H_mosi_seed42_result.md（本文件）
- [x] P6H_mosi_near_miss_analysis.md
- [ ] mosi_metrics_summary.csv（仅摘要指标）
- [ ] 预测/权重/曲线等 CSV/PNG（需修改训练脚本保存 per-sample 输出）

---

## 8. 缺失产物说明

以下产物因训练脚本未保存中间数据而无法生成：

| 文件 | 原因 |
|---|---|
| mosi_predictions_test.csv | 脚本只 print 不 save |
| mosi_awaf_weights_test.csv | 脚本只 print 不 save |
| mosi_text_base_delta_test.csv | 脚本不导出 per-sample delta |
| mosi_group_error_analysis.csv | 需 per-sample prediction |
| mosi_training_curves.png | 无 per-epoch 日志 |
| mosi_confusion_matrix.png | 需 per-sample prediction |
| mosi_awaf_weight_distribution.png | 需 per-sample weights |
| mosi_gate_distribution.png | 需 per-sample gate |
| mosi_delta_distribution.png | 需 per-sample delta |
| mosi_text_vs_final_scatter.png | 需 per-sample text_base & final |

**建议**：在下一轮训练前修改脚本，增加 per-sample 输出保存和 per-epoch metrics CSV。
