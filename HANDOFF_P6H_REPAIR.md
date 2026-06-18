# HANDOFF P6H-R — AWAF/Delta/Gate 残差修复完成

**Date**: 2026-06-18  
**Branch**: `p6h-lora-env-fix-main-training-baseline-smoke`  
**Commit**: `a862dcc` (unchanged — 未提交修复代码)

---

## 本阶段完成

1. ✅ Task 0: 状态确认 — text_base 健康但残差无效，AWAF 崩溃
2. ✅ Task 1: 输出保存系统 — 新建 `scripts/train_textft_lora_mainline.py` (per-sample + per-epoch + 6 图)
3. ✅ Task 2: 模型修复开关 — 修改 `models/fusion/awaf.py` (+LayerNorm/entropy/uniform_mix), 新建 `models/textft_lora_xlstm_awaf_residual.py`
4. ✅ Task 3: R1-R4 sweep (val only) — 4 配置全部完成
5. ✅ Task 4: Best config (R2) full seed42 + test
6. ✅ Task 5: 决策 — Case D (修复失败)
7. ⬜ Task 6: Git (未提交，等用户确认)

---

## 关键发现

### AWAF 权重修复成功 ✅
- P6H: w_v=98.4% 崩溃
- P6H-R R2: w_t=26.3%, w_a=22.8%, w_v=51.0% ✅
- λ_entropy=0.01 成功防止权重崩溃

### Residual 仍然无效 ❌
- 6 次训练 (P6H + R1-R4 + full R2) 全部 residual_gain = 0.00%
- Gate 始终关闭 (0.30-0.51)
- Delta 始终极小 (0.005-0.05)

### Text_base 退化 ❌
- P6H: 86.43% (lr=5e-5, 50ep)
- P6H-R: 81.86% (lr_lora=3e-6, 42ep)
- 低 LoRA lr 导致文本欠训

### Val/test 严重不匹配 ⚠️
- best_val_ACC2 = 88.89%
- test_ACC2 = 81.86%
- gap = 7.03% → val (229 samples) 过小，模型过拟合

---

## 新增/修改文件

### 新建
- `models/textft_lora_xlstm_awaf_residual.py` — 完整主模型类 (repair switches)
- `scripts/train_textft_lora_mainline.py` — 完整训练脚本 (config + outputs)
- `configs/experiments/p6h_repair/mosi_r1_norm_tau_dsr_gate.yaml`
- `configs/experiments/p6h_repair/mosi_r2_entropy.yaml`
- `configs/experiments/p6h_repair/mosi_r3_uniform_mix.yaml`
- `configs/experiments/p6h_repair/mosi_r4_conservative.yaml`
- `configs/models/textft_lora_awaf_mosi_p6h_repair_best.yaml`
- `reports/P6H_lora_env_main/repair/P6H_R_state_confirmation.md`
- `reports/P6H_lora_env_main/repair/P6H_R_model_patch.md`
- `reports/P6H_lora_env_main/repair/P6H_R_repair_sweep.md`
- `reports/P6H_lora_env_main/repair/P6H_R_repair_sweep.csv`
- `reports/P6H_lora_env_main/repair/P6H_R_mosi_repair_best_seed42_result.md`
- `reports/P6H_lora_env_main/repair/P6H_R_residual_still_failed.md`
- `HANDOFF_P6H_REPAIR.md` (本文件)

### 修改
- `models/fusion/awaf.py` — +use_modal_layernorm, +awaf_uniform_mix, +return_diagnostics, +compute_entropy
- `reports/experiment_registry.csv` — +5 条实验记录

### 输出产物 (outputs/P6H_repair/)
- R1-R4 sweep outputs
- BEST_R2_full: result.json, predictions_test.csv, awaf_weights_test.csv, text_base_delta_test.csv, group_error_analysis.csv, gate_delta_stats.csv, 6 PNGs

---

## 下一轮结构救援建议 (详见 P6H_R_residual_still_failed.md)

**优先级**:
1. P0: 提升 LoRA lr 回 5e-5 → 恢复 text_base 至 86%+
2. P0: 验证 audio/vision 单模态 ACC2 → 确认特征含信号
3. P1: 尝试 detach text_base → 强制 residual 学习
4. P1: 或设计 text-confidence conditioned residual

---

## 无法完成或仍需确认

| # | 问题 |
|---|---|
| Q1 | 是否同意将修复代码提交到新分支 `p6h-r-awaf-delta-gate-repair`？ |
| Q2 | 是否同意下一轮先恢复 LoRA lr 然后执行结构救援方案？ |
| Q3 | 是否需要先验证 audio/vision 单模态特征质量？ |
| Q4 | 是否允许对 MOSI val set 过小 (229 samples) 导致的过拟合采取对策（如用 k-fold 或增大 val）？ |

---

## 核心指标总结

| Run | text_base | final | gain | AWAF (t/a/v) |
|---|---|---|---|---|
| P6H original | 86.43% | 86.43% | 0.00% | .006/.010/.984 |
| P6H-R R2 best | 81.86% | 81.86% | 0.00% | .263/.228/.510 |
