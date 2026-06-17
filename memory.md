# memory.md：多模态情感分析项目阶段记忆

> 项目根路径（固定）：`E:\00project_code\main_leo\new_code`  
> 用途：记录 CC/Codex 的阶段性事实、决策、修改、实验结果和风险。  
> 当前状态：旧 handoff 已删除，P0 待重新扫描生成。

---

## 0. 当前稳定事实

### 0.1 论文题目

《基于扩展 LSTM 与 Transformer 的跨模态情感分析研究》

### 0.2 项目根路径

```text
E:\00project_code\main_leo\new_code
```

### 0.3 只读参考路径

```text
E:\00project_code\main_leo\mme\Tri_modal_ER
E:\00project_code\main_leo\refer
```

### 0.4 主模型当前原则

最低主干：三模态投影 + 三路 sLSTM + AWAF（二阶交互）+ 多任务预测头。

DEConv、Data2Vec-Audio、CME 是候选增强模块，必须通过真实数据实验裁决是否进入最终主模型。

### 0.5 旧架构图状态

旧主模型架构图作废，不得在代码实现、重构、论文绘图中参考。最终架构图根据冻结后的真实代码重新绘制。

### 0.6 代码策略

必要时直接重写代码，而不是围绕不可运行或不匹配的旧代码打补丁。

### 0.7 输出规范

代码关键处中文注释；图像 ≥900 dpi；表格必须详细并输出 xlsx/docx；所有实验登记到 `reports/experiment_registry.csv`。

---

## 1. 阶段记录模板

```markdown
## YYYY-MM-DD PXX：阶段名称

### 目标

### 实际完成

### 关键发现

### 修改/新增文件
| 文件路径 | 操作 | 原因 | 是否备份 | 是否影响原逻辑 |
|---|---|---|---|---|

### 实验运行
| 实验编号 | 模型 | 数据集 | 命令 | 是否成功 | 输出路径 | 备注 |
|---|---|---|---|---|---|---|

### 核心结果
| 模型 | 数据集 | seed | ACC2_Non0 | F1_Non0 | ACC2_Has0 | F1_Has0 | MAE | Corr | ACC7 | 可信度 | 是否入论文 |
|---|---|---|---|---|---|---|---|---|---|---|---|

### 问题与风险

### 对论文影响

### 待用户/网页版 AI 判断

### 下一步
```

---

## 2. P0 阶段记录 (2026-06-16)

### 目标
只读扫描项目全貌，不修改任何代码，不训练，不下载，不安装。

### 实际完成
- 读取全部 11 个 md 配置文件 + 4 个旧结果文件（只记录，不采用）
- 扫描全部目录：根目录、utils_models、utils_tools、utils_train、tmdc_adapter、experiments、data、results_20260307、创新点对应论文、logs
- 审计 5 个历史代码文件（lstm_v.py, ours_model.py, DEConv.py, attention_encoder.py, metricsTop.py）
- 确认数据状态（原始数据存在，无预处理 pkl 特征）
- 确认 baseline 状态（全部 7 个 baseline 代码均不在 new_code）
- 确认环境状态（RTX 5070 Ti 16GB, mme env PyTorch 2.3.0, 需新建 mme_xlstm）
- 确认论文初稿存在（3MB, 2025-03-18）

### 关键发现
1. `lstm_v.py` 是 mLSTM（非 sLSTM），许可证 AGPL-3.0，不可复用 — 必须 P2 自实现 sLSTM
2. 当前 `model.py`(MainModelV9) 使用 Transformer+GatedFusion，与目标架构完全不兼容 — 必须 P2 重写
3. `new_code/data` 中无可用预提取特征 pkl — P1 必须建立特征链路
4. `metricsTop.py` 缺少 Non0 口径 — 必须重写
5. 旧 Tri_modal_ER 中 MOSI 特征文件名含 "simulated" — 需 P1 审计
6. 无 AWAF 实现、无 sLSTM 实现、无统一 metrics、无 configs/ 目录
7. 旧架构图已作废，旧 handoff 已删除

### 修改/新增文件
| 文件 | 操作 | 原因 |
|------|------|------|
| reports/P0_status_scan.md | 新建 | P0 完整扫描报告 |
| reports/P0_initial_file_structure.md | 新建 | 初始文件结构记录 |
| reports/P0_environment_plan.md | 新建 | 环境审计与规划 |
| reports/P1_feature_decision_plan.md | 新建 | P1 特征裁决计划 |
| reports/P2_main_backbone_plan.md | 新建 | P2 主模型实现计划 |
| reports/P3_candidate_module_selection_plan.md | 新建 | P3 候选模块裁决计划 |
| HANDOFF_PHASE_00.md | 新建 | P0 阶段 handoff |
| docs/DECISIONS.md | 新建 | 关键决策记录 (D001-D005) |
| memory.md | 追加 | P0 阶段记录 |

### 需用户/网页版 AI 判断
见 HANDOFF_PHASE_00.md 第 9 节 (Q1-Q6)

### 下一步
等待用户和网页版 AI 审阅 HANDOFF_PHASE_00.md，确认后进入 P1（特征链路裁决）。

---

## 3. P1 阶段记录 (2026-06-16)

### 目标
特征链路裁决，确定 MOSI/MOSEI 后续统一使用的三模态特征方案。

### 实际完成
- 网络核验 MLCL 仓库（URL 确认但内容被网络限制阻止）
- 网络核验 CASP 仓库（backbone 可分离，Python≥3.8, PyTorch≥1.8.0）
- TMDC MOSI 特征审计（2199 样本, 1024d×3, 100% 标签对齐, 真实提取 ✅）
- Simulated MOSI 特征审计（放弃 ❌）
- 旧 MOSEI 特征审计（标签 0-1 错误 + 维度不一致 ❌）
- MOSEI TMDC 自建成本评估（Text+Audio 可提取，Vision 缺原始视频帧）
- Python 3.10 推荐 + 环境规划

### 关键发现
1. TMDC MOSI 特征真实可用，标签与官方 100% 对齐 → 可直接支撑 P2 smoke test
2. MOSEI 无可用特征，需自建 TMDC 链路（全部提取脚本为 MOSI 专用）
3. MLCL 仓库内容无法通过网络确认（需用户浏览器直接访问）
4. CASP backbone 可分离，P6 普通 baseline 只使用 pretrain backbone
5. 全部旧特征链路不可用 → 统一采用 TMDC 自建方案

### 特征方案裁决
- ✅ 主推荐：方案 B (TMDC 自建) — MOSI 已就绪，MOSEI 需自建
- ⚠ 备用：方案 A (MLCL 标准) — 待用户确认仓库内容
- ❌ 不采用：方案 C (旧特征) — 全部不可用

### 修改/新增文件
| 文件 | 操作 |
|------|------|
| reports/P1_mlcl_casp_web_audit.md | 新建 |
| reports/P1_local_feature_audit.md | 新建 |
| reports/P1_feature_plan.md | 新建 |
| reports/P1_environment_compatibility.md | 新建 |
| data/README_data.md | 新建 |
| HANDOFF_PHASE_01.md | 新建 |
| docs/DECISIONS.md | 追加 D006-D010 |
| memory.md | 追加 P1 记录 |

### 需用户/网页版 AI 判断
见 HANDOFF_PHASE_01.md Q1-Q4

### 下一步
等用户确认后进入 P2（最低主干实现）。

---

## 5. P2 阶段记录 (2026-06-16)

### 目标
创建 mme_xlstm 环境，实现最低主干代码（sLSTM + AWAF + 多任务头），MOSI 真实数据 1 epoch smoke test。

### 实际完成
- 决策记录 D011-D015 (MLCL下载授权、MOSEI视觉、mme_xlstm、CLIP标注、T=1科学性边界)
- mme_xlstm 环境创建 (Python 3.10.20, PyTorch 2.5.1+cu124)
- 实现 utils/metrics.py (7 指标, 边界测试通过)
- 实现 utils/seed.py (random/numpy/torch/cuda 统一控制)
- 实现 models/encoders/slstm.py (SLSTMCell + SLSTMEncoder, 纯 PyTorch, ~500行)
- 实现 models/fusion/awaf.py (AdaptiveWeightedAttentionFusion, 7种融合模式, ~400行)
- 实现 models/heads.py (RegressionHead + ClassificationHead + UnimodalHeads)
- 实现 models/ours_xlstm_fusion.py (主模型, ~250行, 3.16M params)
- 实现 data/dataset.py (TMDCMOSIDataset, 兼容未来 [T,D] 序列)
- 实现 engine/trainer.py (训练+评估+产物保存)
- 实现 configs/default.yaml + configs/models/ours_backbone.yaml
- 实现 scripts/smoke_forward.py + scripts/train_main.py
- 随机张量 forward test 全部通过 (5/5)
- MOSI TMDC 真实数据 1 epoch smoke test 通过 (T=1, CPU模式)
- env/install_commands.md + pip_freeze 完成

### 关键发现
1. RTX 5070 Ti (Blackwell sm_120) 需要 CUDA 12.4+ → PyTorch 2.5.1+cu124
2. 清华镜像 SSL 证书问题 → 切换默认 PyPI
3. AWAF sum(w)=1 max_dev=1.19e-7 ✅
4. 1 epoch 后 AWAF 权重：text=0.223, audio=0.207, vision=0.570 (Vision 在 T=1 下占主导)
5. 模型 3.16M 参数，前向/反传正常

### 修改/新增文件 (23个)
核心代码 (10个): metrics.py, seed.py, slstm.py, awaf.py, heads.py, ours_xlstm_fusion.py, dataset.py, trainer.py, default.yaml, ours_backbone.yaml
脚本+配置 (5个): smoke_forward.py, train_main.py, install_commands.md, pip_freeze, __init__.py×5
报告 (4个): P2_main_model_smoke.md, P2_environment_setup.md, HANDOFF_PHASE_02.md
更新 (3个): memory.md, DECISIONS.md (D011-D015), data/README_data.md

### 待完成
- GPU smoke test ✅ (PyTorch nightly 2.12.0.dev+cu128 已安装)
- env/environment_mme_xlstm.yml (conda env export)
- MLCL 下载审计 (可选，P2不阻塞)

### P2.1 质量闸门 (2026-06-16)
- sLSTM mask 状态冻结修复 ✅ (padding 噪声不影响 pooled)
- AWAF modality dropout 逐样本修复 ✅ (per-sample at-least-one-modal)
- AWAF context self-exclusion 修复 ✅ (q→other 2 modals)
- 单元测试全部通过 ✅
- GPU smoke test 重跑通过 ✅
- 决策 D016-D018 记录 ✅

### P3A C0 稳定性验证 ✅ + P3B 候选模块裁决 ✅ (2026-06-16)

- Stable PyTorch 2.11.0+cu128 环境成功
- P3A: 14 runs, C0 best config: h256-l1-lr1e-4-d0.3
- P3B: 6 runs, DEConv (+1.07%) 推荐 P4, CME 不稳定 (42:+2.1% 2024:-0.5%)
- C2 (Data2Vec-Audio) 暂缓

---

### P3C 最终冻结诊断 (2026-06-16)
- 40ep 揭示了关键反转：C1/C3 均劣于 C0
- P3B 20ep 结论被推翻 — C1 在 40ep 退化 2%
- P4 主模型 = C0 (sLSTM+AWAF only)
- DEConv/CME/Data2Vec 均不进入 P4 主模型

### P4A + P4R 审计 (2026-06-16)
- P4A MOSI: ACC2_NZ=75.25% (3 seeds, 60ep)
- P4R CRITICAL: test leakage found (best_epoch by test, not val)
- P4R: ACC2 using reg_sign is +1.4% higher than cls_logit (MMSA convention)
- P4R: T=1 features are root cause of low performance vs MMSA ~85%
- P4R: Recommend Route B — switch to sequence features (T~50)

### P4S + P4T (2026-06-16/17)
- MLCL seq (T~12): ACC2_NZ=44.9% ❌ (weak features)
- TMDC T=1 pooled: ACC2_NZ=75.25% ✅ (strong pretrained)
- P4R: test leakage found, protocol fixed
- P4T: cleanup classified (no deletion), GitHub prep done
- Strong seq extraction: pipeline ready, models downloading

### P4U (2026-06-17) — Cleanup, Git prep
- Deleted old V9 code, results, caches. All integrity checks passed.
- Branch: p4u-clean-before-architecture-upgrade, commit 08c67fa

### P4U.1 (2026-06-17) — Repo fix, mask bug, configs
- Fixed .gitignore, strict_trainer mask, check_val API
- Branch: p4u1-fix-repo-data-and-mask, commit b4ac03d

### P4V (2026-06-17) — AWAF-Seq Architecture
- AWAF-Seq modules: CrossModalTransformer + MaskedAttentionPooling + OursAWAFSeqXLSTM
- All tests pass (3/3). Branch: p5v-awafseq-crossmodal-upgrade, commit 73f984a.

### P4V.1 (2026-06-17) — AWAF-Seq 73.9%. Branch: p4v1-awafseq-training-probe.

### P4W (2026-06-17) — MOSI 78.4%. Branch: p4w-mosi-awafseq-performance.

### P4X (2026-06-17) — MOSI final + MOSEI prep. Branch: p4x-mosi-vision-mosei-prep.

### P4Y (2026-06-17) — Data Audit
- MOSI: Grade B+, 2199 MP4+WAV+labels → KEEP (rare, enables Vision T>1)
- MOSEI: Grade C→B, NO MP4 available (CMU YouTube privacy)
- Old MOSEI .features deleted (1.26GB unreliable). SDK .csd is standard approach.
- Branch: p4y-data-redownload-audit, commit d0f2b04 ✅

### P5A (2026-06-17) — Text-Guided AWAF-xLSTM (FAILED)
- ACC2_NZ_reg=75.5%, MAE=1.053, Corr=0.611 — worse than P4W 78.8%
- Text-guided mechanism broke audio/vision independent sLSTM modeling
- Still inherited text-sLSTM from old route → unavoidable degradation

### P5B (2026-06-17) — Strong Feature Upper Bound Diagnosis (BREAKTHROUGH)
- Text+sLSTM=76.2% vs DeepMLP text-only=80.2%: sLSTM on text is COUNTERPRODUCTIVE
- DeepMLP text-only (592K params) BEATS Full AWAF-Seq (4.15M params): 80.2% > 78.8%
- Multimodal fusion currently HURTS, not helps
- Root cause: DeBERTa tokens already contextualized → sLSTM adds noise
- HANDOFF_PHASE_10B.md documents the breakthrough

### P5C (2026-06-17) — DeepText-xLSTM-AWAF Residual Refactor ✅
- **D031**: Architecture overhaul — main model redesigned as DeepText-xLSTM-AWAF Residual
- **Unit tests**: 9/9 passed (MOSI dims, MOSEI SDK dims, AWAF sum=1, padding, ablations, delta_scale, loss)
- **3ep smoke**: ACC2=75.00% (trending up strongly)
- **Seed42 60ep**: **ACC2_NZ_reg=81.10%, MAE=0.8155, Corr=0.7487** ✅
  - Beats DeepMLP text-only (80.2%) by +0.9% — residual fusion VALIDATED
  - Beats P4W AWAF-Seq (78.8%) by +2.3%
  - Best val ACC2=84.26% at epoch 51
  - AWAF weights: w_t=0.18, w_a=0.44, w_v=0.39 — audio dominates residual
- **Decision**: Model validated. Recommend seed=2024 confirmation. Model NOT frozen.
- Branch: p5c-deeptext-xlstm-awaf-residual-refactor
- Commit: (pending)
