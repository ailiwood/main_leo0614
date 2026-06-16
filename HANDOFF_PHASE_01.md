# HANDOFF_PHASE_01.md

> 阶段：P1 特征链路裁决  
> 生成时间：2026-06-16  
> 状态：✅ P1 完成，待用户和网页版 AI 审阅

---

## 1. 本阶段读过哪些文件

| 文件 | 状态 | 作用 |
|------|:--:|------|
| claude.md | ✅ | 重新确认 baseline 口径和特征要求 |
| workflow.md | ✅ | 确认 P1 任务范围和权限 |
| 实验设计.md | ✅ | 确认特征方案 A/B/C 设计要求 |
| 主模型实现规划.md | ✅ | 确认主模型输入维度要求 |
| 代码重构建议.md | ✅ | 确认 data/ 目标结构 |
| P0_status_scan.md | ✅ | 参考 P0 发现的风险 |
| P0_initial_file_structure.md | ✅ | 参考文件结构 |
| P0_environment_plan.md | ✅ | 参考环境规划 |
| P1_feature_decision_plan.md | ✅ | 按计划执行 |
| DECISIONS.md | ✅ | 参考已有决策 |
| memory.md | ✅ | 参考阶段记忆 |
| 经验总结.md | ✅ | 参考避坑规则 |

---

## 2. 本阶段新增/修改文件

| 文件 | 操作 | 内容 |
|------|:--:|------|
| `reports/P1_mlcl_casp_web_audit.md` | 新建 | MLCL/CASP 网络核验报告 |
| `reports/P1_local_feature_audit.md` | 新建 | TMDC/simulated/MOSEI 本地特征审计 |
| `reports/P1_feature_plan.md` | 新建 | ★ 三方案比较与最终裁决 |
| `reports/P1_environment_compatibility.md` | 新建 | Python 3.10 推荐 + 环境规划 |
| `data/README_data.md` | 新建 | ★ 特征版本固化说明 |
| `HANDOFF_PHASE_01.md` | 新建 | P1 阶段 handoff |
| `docs/DECISIONS.md` | 追加 | D006-D010 决策记录 |
| `memory.md` | 追加 | P1 阶段记录 |

**未修改任何已有代码、数据、模型、论文或配置文件。未运行训练、提取、安装。**

---

## 3. 核心结论

### 3.1 MLCL 特征

**⚠️ 无法通过网络直接确认可用性。** GitHub 直接抓取和 raw 内容获取均被网络策略阻止。WebSearch 对仓库的索引极不完整。

**待用户确认**（浏览器直接访问 https://github.com/YetZzzzzz/MLCL）：
- 是否有 MOSI/MOSEI 标准特征下载链接
- 特征格式、维度、split
- 许可证

### 3.2 TMDC MOSI 特征

**✅ 真实、可用、100%标签对齐。**

- 2199 clips, 93 videos, 标准 MOSI split (1284/229/686)
- Text(DeBERTa-1024d) + Audio(wav2vec2-1024d) + Vision(CLIP→1024d)
- 可直接支撑 P2 MOSI smoke test

### 3.3 TMDC MOSEI 特征

**❌ 未提取。需要自建提取链路。**

- Audio_chunk wav 文件存在（5245 个）
- Labels CSV 存在（含 sentiment -3~+3 + text）
- 所有 TMDC 提取脚本为 MOSI 专用
- 视觉特征提取缺少原始视频/帧来源

### 3.4 旧特征

全部不可用：simulated（放弃）、Tri_modal_ER MOSEI（标签错误+维度不一致）、V9 RoBERTa（路线排除）。

---

## 4. 特征方案裁决

| 方案 | 裁决 |
|------|------|
| **方案 B**（TMDC 自建） | ✅ **主推荐** — MOSI 已就绪，MOSEI 需自建 |
| 方案 A（MLCL 标准） | ⚠ 备用 — 待用户确认仓库内容 |
| 方案 C（旧特征） | ❌ 不采用 — 全部不可用 |

**具体**：
- P2 smoke test：使用 TMDC MOSI 特征
- P4 MOSEI 训练：需先自建 MOSEI TMDC 特征
- P6 MLCL baseline：若 MLCL 特征可用，用于 MLCL baseline；主模型用 TMDC 特征（论文中标注差异）

---

## 5. P2 smoke test 就绪状态

| 数据集 | 就绪 | 方案 |
|------|:--:|------|
| MOSI | ✅ | TMDC 特征直接使用 |
| MOSEI | ❌ | 等 P2 期间自建 |

**P2 策略**：先用 MOSI 做全流程 smoke test（模型→训练→评估），MOSEI 提取并行进行。

---

## 6. 需要用户/网页版 AI 判断

### Q1（紧急）：MLCL 仓库确认
请用户通过浏览器访问 https://github.com/YetZzzzzz/MLCL，截图或复制 README 中的数据下载说明。如果提供特征下载，是否授权下载到 `external/MLCL_features/`？

### Q2（紧急）：MOSEI 视觉特征来源
MOSEI 没有预提取视频帧。需要确认：
- 选项1：用户提供原始 MOSEI 视频路径（目前 `data/CMU-MOSEI` 的 Test_original/Val_original 中有什么？）
- 选项2：使用公开的 MOSEI 视觉特征（如 MMSA 框架提供的 FACET 特征）
- 选项3：先不提取视觉特征，仅用 Text+Audio 双模态跑 MOSEI（损失可接受吗？）

### Q3（常规）：mme_xlstm 环境创建时机
Python 3.10 推荐已给出。是否同意 P2 开始前创建 mme_xlstm 环境并安装核心依赖？

### Q4（常规）：TMDC MOSI 视觉特征
MANet 命名实际使用的是 CLIP-ViT-B/32 特征（768d→random proj→1024d）。在 data/README_data.md 中是否按真实情况标注为 "CLIP-ViT-B/32" 而非 "MANet"？

---

## 7. P2 下一步建议

```
P2（最低主干实现）:
  1. 创建 mme_xlstm 环境（如 Q3 确认）
  2. 实现 utils/metrics.py（统一指标）
  3. 实现 models/encoders/slstm.py（纯 PyTorch sLSTM）
  4. 实现 models/fusion/awaf.py（AWAF 含8种消融开关）
  5. 实现 models/heads.py + models/ours_xlstm_fusion.py
  6. 实现 data/dataset.py（从 TMDC 特征加载三模态数据）
  7. 实现 engine/trainer.py（最小版）
  8. configs/default.yaml
  9. MOSI 随机张量 forward test → MOSI 真实数据 1 epoch smoke test
  10. 并行：评估/启动 MOSEI 特征提取
```

---

## 8. 明确声明

本阶段（P1）：
- ✅ 已通过网络搜索核验 MLCL 和 CASP
- ✅ 已审计 TMDC MOSI 特征（2199样本，真实提取，100%标签对齐）
- ✅ 已审计模拟特征和旧 MOSEI 特征（均为不可用）
- ✅ 已评估 MOSEI 自建方案和成本
- ✅ 已形成特征方案裁决（推荐 TMDC 自建）
- ✅ 已完成环境兼容性分析（推荐 Python 3.10）
- ✅ 已输出 6 个报告/计划 + data README + HANDOFF

本阶段：
- ❌ 没有运行任何训练
- ❌ 没有运行任何特征提取
- ❌ 没有下载任何数据或代码
- ❌ 没有修改任何已有文件
- ❌ 没有创建任何新环境或安装任何包
- ❌ 没有使用 simulated 特征作为正式实验方案
- ❌ 没有恢复 V9 RoBERTa 特征链路
- ❌ 没有进入 P2 编码实现
