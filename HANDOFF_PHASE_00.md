# HANDOFF_PHASE_00.md

> 阶段：P0 初始化扫描  
> 生成时间：2026-06-16  
> 状态：✅ P0 完成，待用户和网页版 AI 审阅

---

## 1. 本阶段读过哪些配置文件

| 文件 | 状态 |
|------|:--:|
| claude.md | ✅ 完整阅读 |
| workflow.md | ✅ 完整阅读 |
| 实验设计.md | ✅ 完整阅读 |
| 主模型实现规划.md | ✅ 完整阅读 |
| 代码重构建议.md | ✅ 完整阅读 |
| 最新学术进展与代码推荐.md | ✅ 完整阅读 |
| memory.md | ✅ 完整阅读 |
| 经验总结.md | ✅ 完整阅读 |
| project_instruction.md | ✅ 完整阅读 |
| CC启动提示词_两阶段版.md | ✅ 完整阅读 |
| 0613实验FINAL_REPORT.md | ✅ 只记录，不采用 |
| results_20260307/*.md (4个) | ✅ 只记录，不采用 |

---

## 2. 扫描过哪些目录

| 目录 | 内容概要 |
|------|----------|
| `new_code/` 根目录 | 10 md + 6 py + 1 docx |
| `utils_models/` | 6 py (lstm_v.py 等) |
| `utils_tools/` | 3 py (metricsTop.py 等) |
| `utils_train/` | 1 py (en_train.py) |
| `tmdc_adapter/` | 10 py + features/ + hf_cache/ |
| `experiments/v9_roberta*/` | 5 pth 权重 + 4 json |
| `data/mosi/` | Frames/ + Raw/ + wav/ + label.csv |
| `data/CMU-MOSEI/` | Audio_chunk/ + Labels/ + Test/Val_original |
| `results_20260307/` | 4 md + 4 图像 + 2 json |
| `创新点对应论文/` | 2 PDF |
| `logs/` | 2 log |

---

## 3. 当前代码结构概览

**核心发现：当前代码全为 V9 RoBERTa 路线，与目标主模型完全不兼容。**

- 当前模型：`MainModelV9` = TransformerEncoder ×3 + GatedFusion（无 sLSTM/AWAF/xLSTM/CME/DEConv）
- 当前训练入口：`train.py`，硬编码旧项目路径，仅 MOSEI
- 历史代码：`utils_models/` 中的 `lstm_v.py`(mLSTM+AGPL-3.0), `ours_model.py`(旧架构), `DEConv.py`, `attention_encoder.py`(vision_xLSTM+CME)
- 完全缺失：`configs/`, `models/`, `engine/`, `scripts/`, `utils/metrics.py`

详见：`reports/P0_initial_file_structure.md`

---

## 4. 数据 / 特征现状

- **原始数据**：MOSI (93视频) 和 MOSEI (Audio_chunk) 均存在
- **预提取特征**：`new_code/data` 中**无**可直接使用的三模态特征 pkl 文件
- **TMDC 特征**：MOSI 已提取 (DeBERTa+wav2vec+MANet)，MOSEI 未提取（目录为空）
- **旧特征**：V9 使用的 RoBERTa-large pkl 在外部项目路径 (`D:\business\...`)
- **MOSI 旧特征标记为 "simulated"**（`mosi_simulated_features.pkl`），真实性存疑

---

## 5. 历史模型与旧代码审计结果

| 文件 | 关键结论 |
|------|----------|
| `lstm_v.py` | mLSTM (非 sLSTM), AGPL-3.0 ❌ 不可复用 |
| `attention_encoder.py` | 依赖 AGPL lstm_v.py ❌ 不可直接复用 |
| `DEConv.py` | 自定义 DEConv_2, 可参考但需适配 |
| `ours_model.py` | 旧架构 (RoBERTa+Data2Vec+DEConv+CME) ≠ 目标 |
| `metricsTop.py` | 缺少 Non0 口径 ❌ 必须重写 |
| `model.py` (MainModelV9) | Transformer+GatedFusion ❌ 与目标完全不兼容 |

---

## 6. baseline 状态

**所有 7 个 baseline（TFN/LMF/MulT/MISA/Self-MM/MLCL/CASP）代码均不在 `new_code` 中。**

需在 P6 统一接入。0613 报告中的 MLCL 84.00% 基准待核实是否与 YetZzzzzz/MLCL (TMM 2025) 同源。

---

## 7. 环境状态

- GPU: RTX 5070 Ti, 16GB VRAM ✅
- Conda: E:\Anaconda3, mme 环境 (Python 3.9.21, PyTorch 2.3.0+cu118, CUDA available)
- 推荐新建: mme_xlstm (E:\Anaconda3\envs\mme_xlstm)
- 暂无 mme_xlstm 环境

详见：`reports/P0_environment_plan.md`

---

## 8. 主要风险

| # | 风险 | 严重程度 | 建议 |
|---|------|:--:|------|
| 1 | 无可用的三模态特征 pkl，数据链路需从头建立 | 🔴 高 | P1 优先解决 |
| 2 | lstm_v.py 是 mLSTM+AGPL-3.0，无法直接复用 | 🔴 高 | P2 自实现 sLSTM |
| 3 | 当前代码全为 V9 路线，与目标不兼容 | 🔴 高 | P2 重写核心代码 |
| 4 | MOSI 旧特征存在 "simulated" 标记 | 🔴 高 | P1 审计确认 |
| 5 | 无统一 metrics.py | 🟡 中 | P2 首先实现 |
| 6 | 所有 baseline 代码缺失 | 🟡 中 | P6 接入 |
| 7 | MOSEI TMDC 特征未提取 | 🟡 中 | P1 评估 |
| 8 | 论文初稿可能包含旧结论/旧指标 | 🟡 中 | P9 修正 |

---

## 9. 需要用户/网页版 AI 判断的问题

### Q1：P1 特征来源决策
是否允许 P1 联网检索 MLCL GitHub 仓库（仅检索 README，不下载），以评估是否可复用 MLCl 标准特征？如果 MLCL 特征可用，是否授权下载？

### Q2：MOSI "simulated" 特征
`Tri_modal_ER/data/mosi_simulated_features.pkl` 文件名含 "simulated"。P1 是否应审计此文件并确认其真实性？如果确实是模拟数据，则 MOSI 必须从原始数据重新提取或复用 MLCL 标准特征。

### Q3：sLSTM 自实现 vs 官方库
P2 将纯 PyTorch 自实现 sLSTM。是否需要先阅读 NX-AI 官方 sLSTM 论文公式（不从 AGPL 代码复制），以确保公式准确？

### Q4：mme_xlstm 环境创建时机
建议 P2 前创建新环境 `mme_xlstm`。是否同意？Python 版本建议 3.10 还是 3.11？

### Q5：CASP baseline 的 TTA 属性
CASP 是 test-time adaptation 方法。P6 接入时的评估策略是否需特殊处理（如不使用其 adaptation 步骤，仅用其 backbone）？

### Q6：论文初稿风险预判
论文初稿（2025-03-18）可能已将 V9 路线写为 xLSTM-Fusion、将普通门控融合写为 AWAF。P9 重写前是否需要提前告知导师这一风险？

---

## 10. P1 / P2 / P3 下一步建议

```
P1（特征链路裁决）:
  1. 检索 MLCL 标准特征可用性（只读检索）
  2. 审计 TMDC MOSI 特征和 "simulated" 标记
  3. 评估 MOSEI 特征提取方案
  4. 输出 data/README_data.md

P2（最低主干实现）:
  1. 先写 utils/metrics.py 和 utils/seed.py
  2. 实现 models/encoders/slstm.py（纯 PyTorch 自实现）
  3. 实现 models/fusion/awaf.py（含8种消融开关）
  4. 实现 models/heads.py + models/ours_xlstm_fusion.py
  5. 随机张量 forward test
  6. 等 P1 特征就绪后真实数据 1 epoch smoke test

P3（候选模块裁决实验）:
  C0-C7 实验矩阵，裁决 DEConv/Data2Vec-Audio/CME
```

详见：
- `reports/P1_feature_decision_plan.md`
- `reports/P2_main_backbone_plan.md`
- `reports/P3_candidate_module_selection_plan.md`

---

## 11. 明确声明

本阶段（P0）：
- ✅ 已读取全部 11 个 Markdown 配置文件
- ✅ 已扫描全部代码目录和文件
- ✅ 已审计 5 个历史模型/工具文件
- ✅ 已记录数据、特征、模型权重、实验产物现状
- ✅ 已标记所有风险和待决策事项
- ✅ 已输出 6 个报告/计划 + handoff + 决策记录 + 文件结构记录
- ✅ 已更新 memory.md

本阶段：
- ❌ 没有执行任何训练
- ❌ 没有重构任何核心代码
- ❌ 没有删除任何文件或文件夹
- ❌ 没有修改只读参考路径
- ❌ 没有下载公开代码或模型
- ❌ 没有安装任何新包
- ❌ 没有采用旧主模型架构图
- ❌ 没有把 V9 RoBERTa-large 历史结果写成当前主模型结果
- ❌ 没有把 DEConv/Data2Vec-Audio/CME 写成已确定最终模块
- ❌ 没有把 CASP 当普通端到端 baseline
