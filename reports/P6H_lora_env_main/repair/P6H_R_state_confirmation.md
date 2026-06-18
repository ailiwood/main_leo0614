# P6H-R State Confirmation — MOSI AWAF/Delta/Gate 残差修复

**Date**: 2026-06-18 17:10 UTC+8  
**Branch**: `p6h-lora-env-fix-main-training-baseline-smoke`  
**Commit**: `a862dcc`

---

## 1. 当前 MOSI seed42 结果确认

| 确认项 | 值 | 判断 |
|---|---|---|
| text_base_ACC2_Non0 | 86.43% | ✅ 健康 (>85%) |
| final_ACC2_Non0 | 86.43% | ⚠️ = text_base |
| residual_gain | 0.00% | 🔴 零增益 |
| AWAF w_t | 0.0056 | 🔴 近乎为零 |
| AWAF w_a | 0.0100 | 🔴 近乎为零 |
| AWAF w_v | 0.9844 | 🔴 独占 98.4% |
| gate_mean | 0.4596 | 🔴 抑制 54% 残差 |
| dsr/dsc | 0.02 | 🔴 有效残差 <0.005 |

## 2. 根因确认

| # | 根因 | 严重度 | 确认 |
|---|---|---|---|
| 1 | AWAF 权重崩溃为 vision-only | 🔴 严重 | ✅ 确认 |
| 2 | Delta scale 过小 (0.02) | 🔴 严重 | ✅ 确认 |
| 3 | Gate 抑制 (init_bias=-1.5→sigmoid≈0.18) | 🟡 中等 | ✅ 确认 |
| 4 | 缺少 per-sample/per-epoch 输出 | 🟡 中等 | ✅ 确认 |

## 3. 当前代码结构

| 文件 | 状态 | 说明 |
|---|---|---|
| `models/modules/minimal_lora.py` | ✅ | LoRA 实现，正常 |
| `models/fusion/awaf.py` | ✅ | AWAF 模块，需加 LayerNorm/entropy/uniform_mix |
| `models/modules/uncertainty_residual_gate.py` | ✅ | Gate 模块，init_bias 可配置 |
| `models/encoders/slstm.py` | ✅ | sLSTM 编码器，正常 |
| `models/textft_lora_xlstm_awaf_residual.py` | ❌ 不存在 | **需新建** |
| `scripts/train_textft_lora_mainline.py` | ❌ 不存在 | **需新建** |
| `scripts/p6h_train_lora_main.py` | ✅ | 当前单文件脚本，需替换 |
| `configs/models/textft_lora_awaf_mosi_p6h.yaml` | ❌ 不存在 | **需新建** |

## 4. AWAF 模块当前能力与缺失

| 能力 | 当前状态 |
|---|---|
| tau 可学习 | ✅ 已有 (tau_raw → softplus → tau) |
| 跨模态上下文增强 | ✅ 已有 |
| 二阶 Hadamard 交互 | ✅ 已有 |
| 模态 dropout | ✅ 已有 |
| 模态输入 LayerNorm | ❌ 缺失 |
| AWAF entropy 输出 | ❌ 缺失 |
| Uniform mix (weight floor) | ❌ 缺失 |
| modality_dropout 可选 | ✅ 已有 |

## 5. Gate 模块当前能力

| 参数 | 默认值 | P6H 使用值 | 修复建议 |
|---|---|---|---|
| init_bias | -1.5 | -1.5 (默认) | → +2.0 |
| 初始 sigmoid | ~0.18 | ~0.18 | → ~0.88 |

## 6. 训练脚本当前缺失

| 缺失项 | 影响 |
|---|---|
| per-epoch metrics CSV | 无法绘制训练曲线 |
| per-sample predictions CSV | 无法做 error analysis |
| per-sample AWAF weights CSV | 无法分析权重分布 |
| per-sample delta/gate CSV | 无法诊断残差 |
| 训练曲线 PNG | 缺少可视化 |
| 混淆矩阵 PNG | 缺少可视化 |
| AWAF 分布图 | 缺少可视化 |

## 7. 阶段范围确认

| 确认项 | 判断 |
|---|---|
| 是否只修残差通路？ | ✅ 是 |
| 是否不动 LoRA？ | ✅ 是 |
| 是否不动 tokenizer/label/split？ | ✅ 是 |
| 是否不动 RoBERTa 主干？ | ✅ 是 |
| 是否不启动 MOSEI？ | ✅ 是 |
| 是否不启动 baseline？ | ✅ 是 |
| 是否只用 val 选配置？ | ✅ 是 |
| 是否 test only once at the end？ | ✅ 是 |

## 8. 实施计划

| 任务 | 内容 | 优先级 |
|---|---|---|
| Task 1 | 新建完整模型文件 + 训练脚本 + 输出保存 | P0 |
| Task 2 | AWAF/Delta/Gate 修复开关 | P0 |
| Task 3 | R1-R4 sweep (val only) | P0 |
| Task 4 | Best config full seed42 | P1 |
| Task 5 | 决策分流 | P1 |
| Task 6 | Git & 记录 | P2 |
