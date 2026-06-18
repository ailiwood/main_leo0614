# P6H Resume Status Report — FINAL (MOSI seed42)

**Generated**: 2026-06-18 17:04 UTC+8  
**Status**: ✅ MOSI seed42 训练已完成

---

## 1. 当前环境与仓库

| 项目 | 值 |
|---|---|
| 当前分支 | `p6h-lora-env-fix-main-training-baseline-smoke` |
| 当前 commit | `a862dcc` — "P6H Minimal LoRA device fix" |
| Python | `E:\Anaconda3\envs\mme_xlstm_stable\python.exe` |

---

## 2. MOSI seed42 训练结果

| 指标 | 值 |
|---|---|
| text_base_ACC2_Non0 | 86.43% |
| final_ACC2_Non0 | 86.43% |
| residual_gain | **0.00%** ⚠️ |
| F1_Non0 | 84.36% |
| MAE | 0.6775 |
| Corr | 0.8275 |
| ACC7 | 45.92% |
| best_val_ACC2 | 87.50% |
| AWAF w_t / w_a / w_v | 0.006 / 0.010 / 0.984 ⚠️ |
| gate_mean | 0.46 |

---

## 3. 核心诊断

1. **text_base 健康** (86.43% > 85.37%) → LoRA + RoBERTa 链路正常
2. **AWAF 权重崩溃** → vision 独占 98.4%
3. **Delta scale 过小** → dsr=0.02，有效残差 < 0.005
4. **残差融合零效果** → gain = 0.00%

---

## 4. 分流判断

**介于 Case C/D**：text_base 正常但残差无效。进入调参阶段。

---

## 5. 已生成产物

| 文件 | 状态 |
|---|---|
| `reports/P6H_lora_env_main/mosi/P6H_mosi_seed42_result.md` | ✅ |
| `reports/P6H_lora_env_main/mosi/P6H_mosi_near_miss_analysis.md` | ✅ |
| `reports/P6H_lora_env_main/mosi/mosi_metrics_summary.csv` | ✅ |
| `reports/P6H_lora_env_main/mosi/mosi_metrics_summary.xlsx` | ❌ (no openpyxl) |
| per-sample CSVs / PNGs | ❌ (训练脚本不保存) |

---

## 6. 下一步

等待用户审阅结果与 P6H_mosi_near_miss_analysis.md，确认 P0 修复方向后进入调参。
