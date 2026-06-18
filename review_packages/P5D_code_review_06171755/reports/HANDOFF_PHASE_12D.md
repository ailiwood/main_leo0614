# HANDOFF_PHASE_12D.md — P5D Residual Stability & Performance Sprint

## 本阶段完成

### ✅ 已完成

1. **seed=2024 补跑**：ACC2_NZ=81.40%, MAE=0.7957, Corr=0.7490
2. **2-seed 稳定性验证**：mean ACC2=81.25%, σ=0.15% — 架构稳定
3. **Checkpoint 空间审计**：34 pth 文件, 537MB, 清理计划待用户确认
4. **Residual 效果分析**：
   - weak_neg 最受益（56.2% improved, sign +1.3%）
   - strong_neg 略微被过度修正（53.3% damaged）
   - 整体效果 subtle 但方向正确
5. **ConditionalResidualGate 实现**：8/8 单元测试通过, 75K params, 已集成到主模型
6. **Sample reweight 实现**：weak_neg/weak_pos/near_zero 分组加权 + focal sign loss
7. **Two-stage trainer**：完整脚本 ready (Stage1→Stage2→Stage3)
8. **Residual 分析脚本**：`scripts/analyze_residual_effect.py`
9. **MOSEI 状态维护**：维度兼容 ✅，正式训练仍 blocked

### ⏳ 待运行

| 实验 | 种子 | Epochs | 预计时间 | 优先级 |
|------|------|--------|----------|--------|
| Diagnostic ablation (4 runs) | 42 | 30 | ~45min | High |
| ConditionalResidualGate | 42 | 60 | ~13min | High |
| Two-stage training | 42 | 60 | ~30min | Medium |
| Weak_neg reweight | 42 | 60 | ~13min | Medium |
| Gate + Best combo | 2024 | 60 | ~13min | Low |

## 核心结果

### 2-Seed Summary

| Seed | ACC2_NZ | F1_NZ | MAE | Corr | ACC7 |
|------|---------|-------|-----|------|------|
| 42 | 81.10% | 77.70% | 0.8155 | 0.7487 | 42.13% |
| 2024 | 81.40% | 78.06% | 0.7957 | 0.7490 | 44.61% |
| **Mean** | **81.25%** | **77.88%** | **0.8056** | **0.7489** | **43.37%** |

### 对比基准

| 基准 | ACC2 | 差距 vs Ours |
|------|------|-------------|
| DeepMLP text-only | 80.2% | +1.05% ✅ |
| P4W AWAF-Seq | 78.8% | +2.45% ✅ |
| 83% target | 83.0% | -1.75% ❌ |
| 85% target | 85.0% | -3.75% ❌ |

### Residual Effect (Seed=2024)

| Group | N | Improved% | Sign Δ |
|-------|---|-----------|--------|
| strong_pos | 183 | 55.7% | +0.5% |
| strong_neg | 306 | 46.7% | 0.0% |
| weak_pos | 94 | 51.1% | -1.0% |
| **weak_neg** | 73 | **56.2%** | **+1.3%** |
| near_zero | 106 | 51.9% | 0.0% |

## 判断

| 条件 | 状态 |
|------|------|
| 是否超过 80.2 text-only | ✅ +1.05% |
| 是否稳定超过 P4W | ✅ +2.45% |
| 是否达到 82 | ❌ 81.25% (-0.75%) |
| 是否达到 83 | ❌ |
| 是否达到 85 | ❌ |
| 是否建议 P5E 强特征 | 待 P5D sprint 完成后再判断 |

## 新增/修改文件

### 新增 (10+ 文件)
- `models/modules/conditional_residual_gate.py` — ConditionalResidualGate
- `models/modules/__init__.py`
- `scripts/analyze_residual_effect.py` — Residual analysis
- `scripts/train_deeptext_xlstm_awaf_residual_twostage.py` — Two-stage trainer
- `reports/P5D_residual_sprint/P5D_seed2024_result.md`
- `reports/P5D_residual_sprint/P5D_2seed_summary.csv`
- `reports/P5D_residual_sprint/P5D_checkpoint_space_audit.md`
- `reports/P5D_residual_sprint/P5D_residual_effect_analysis.md`
- `reports/P5D_residual_sprint/P5D_mosei_status.md`
- `reports/P5D_residual_sprint/P5D_stage_decision.md`

### 修改 (6 文件)
- `models/deeptext_xlstm_awaf_residual.py` — 集成 ConditionalResidualGate
- `engine/strict_trainer.py` — text_base_delta CSV + delta_reg_weight + save_last_pth
- `engine/losses.py` — sample reweight + focal sign loss
- `docs/DECISIONS.md` — D032-D038
- `memory.md` — P5D 记录
- (CODE_REVIEW_ENTRYPOINT, 经验总结 — 待更新)

## GitHub

- **分支**: `p5d-residual-stability-performance-sprint`
- **基础**: `p5c-deeptext-xlstm-awaf-residual-refactor`
- **Commit**: (待执行)
- **Remote**: https://github.com/ailiwood/main_leo0614.git

## 仍需确认的问题

1. **Checkpoint 清理**：是否执行删除 30 个探索阶段 pth 文件（~480MB）？
2. **P5D 实验优先级**：是否同意先跑 ConditionalResidualGate（最具潜力）？
3. **P5E 强特征路线**：如果 P5D sprint 仍无法达到 82-83%，是否接受当前结果并转向强特征（wav2vec2-large, better vision）？
4. **MOSEI**：MMSDK 安装阻塞是否值得花时间解决，还是推迟到论文最终阶段？
5. **论文策略**：81.25% 的双 seed 结果 + AWAF 权重的可解释性是否足够支撑论文，还是必须追求更高绝对分数？
