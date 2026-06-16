# HANDOFF_PHASE_03B.md

> 阶段：P3B — 候选增强模块初筛与裁决  
> 日期：2026-06-16  
> 环境：mme_xlstm_stable (PyTorch 2.11.0+cu128 STABLE)

---

## 1. 候选模块实现情况

| 模块 | 实现 | 状态 | 许可证 |
|------|------|:--:|:--:|
| DEConv | `models/enhancements/deconv.py` (DynamicFeatureEnhancer) | ✅ 完成 | 自写 |
| CME | `models/enhancements/cme.py` (LightweightCME) | ✅ 完成 | 自写 |
| Data2Vec-Audio | 模型已加载，特征未提取 | ⏸ 暂缓 | Apache 2.0 |

## 2. 实验汇总 (6 runs × 20 epochs)

| Model | ACC2_NZ | MAE | Corr | vs C0 | Stable? |
|-------|:--:|:--:|:--:|:--:|:--:|
| C0 (baseline) | 74.01% | 1.065 | 0.561 | — | ✅ |
| **C1 (DEConv)** | **75.08%** | 1.056 | 0.565 | **+1.07%** | ✅ |
| C3 (CME) | 74.85% | 1.069 | 0.549 | +0.84% | ❌ |

## 3. 裁决

| 推荐 | 模块 | 理由 |
|:--:|------|------|
| ✅ **P4** | C1 (DEConv) | 两 seed 一致提升, +1.07%, 稳定 |
| ❌ 不进 P4 | C3 (CME) | 不稳定 (42:+2.1%, 2024:-0.5%) |
| ⏸ 暂缓 | C2 (Data2Vec) | 特征未提取 |

## 4. 新增/修改文件

- `models/enhancements/deconv.py` (新建)
- `models/enhancements/cme.py` (新建)
- `models/ours_xlstm_fusion.py` (更新：支持 use_deconv/use_cme)
- `configs/models/ours_c1_deconv.yaml` (新建)
- `configs/models/ours_c3_cme.yaml` (新建)
- `reports/P3B_candidate_module_selection/*.md` (新建)
- `HANDOFF_PHASE_03B.md`

## 5. 下一步建议

**进入 P4：MOSI/MOSEI 正式训练。**

P4 主模型 = C0 + DEConv (C1)。
CME 可保留为 P4 可选消融模块。
Data2Vec-Audio 特征在 P4 前提取后可补做 C2。
C4-C7 组合实验不建议当前进行。

## 6. 仍需确认

- AWAF 文本权重 w_t 在 C1 中升至 0.77 — P4 需监控是否坍缩
- CME 不稳定原因（epoch 不足？数据量小？）— P4 用更多 epoch/seed 再验证
