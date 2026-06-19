# workflow.md：多模态情感分析项目工作流程

> 项目根路径：`E:\00project_code\main_leo`  
> 代码工作根路径：`E:\00project_code\main_leo\new_code`  
> Conda 环境：`mme`  
> 主干顺序：P0 扫描 → P1 特征链路 → P2 主模型 → P3 小样本调参 → P4 全量训练 → P5 重构 → P6 baseline → P7 消融 → P8 表图 → P9 论文 → P10 清理。  

---

## 0. 总体原则

本项目按阶段推进。每阶段必须有输入、动作、输出、验收标准和 handoff。

核心顺序不可颠倒：

```text
先确认数据与特征 → 先实现主模型 → 先小样本调参 → 再全量训练 → 再展开全项目重构和 baseline 接入 → 再做消融 → 最后重写论文。
```

本轮允许并鼓励在必要时直接重新撰写代码，而不是强行修补旧代码。旧代码只能作为参考或迁移来源，不能绑架新主模型设计。

旧主模型架构图作废。代码修改、重构和论文方法设计均不参考该图。最终架构图应在主模型代码稳定后根据真实模块重绘。

---

## 1. P0 初始化与现状扫描

### 目标

只读扫描，不改核心代码，确认项目现状、数据、baseline、旧代码可复用性和风险。

### 必做动作

```bat
cd /d E:\00project_code\main_leo
type claude.md
type workflow.md
type 主模型实现规划.md
type 代码重构建议.md
type memory.md
type 经验总结.md

cd /d E:\00project_code\main_leo\new_code
conda activate mme
python --version
where python
git status --short
```

扫描内容：

1. 本地数据位置、格式、样本数、split 文件、是否已有特征缓存；
2. 是否可以复用 MLCL 仓库提供的 MOSI/MOSEI 标准特征；
3. 当前仓库中所有训练入口、模型文件、指标文件、日志、结果文件；
4. `utils_models/lstm_v.py`、`vision_xLSTM` 等已有 xLSTM 实现，审计其公式、许可证、可复用性；
5. 只读参考目录 `Tri_modal_ER` 中可借鉴但不能直接修改的逻辑；
6. MLCL 和 CASP 的本地代码、版本、运行命令、已复现结果与指标口径；
7. 旧主模型架构图文件位置，并在报告中明确“作废，不作为实现依据”；
8. 论文初稿 docx 是否能正常打开。

### 输出

```text
new_code/reports/P0_status_scan.md
new_code/HANDOFF_PHASE_00.md
new_code/docs/DECISIONS.md（如有关键判断）
memory.md 追加 P0 记录
```

### 验收

P0 不允许正式训练，不允许大规模重构，不允许改核心模型。只允许新增 reports/handoff/docs 类文件。

---

## 2. P1 特征链路与特征复用评估

### 目标

确定 MOSI/MOSEI 三模态输入特征的最终来源与版本。

### 必做动作

1. 优先评估是否复用 MLCL 官方仓库提供的 MOSI/MOSEI 标准特征；
2. 若复用，记录来源、下载命令、维度、序列长度、split、许可证或引用说明；
3. 若无法复用，建立自有特征提取链路；
4. 输出统一 dataset/dataloader 读取规范；
5. 写入 `data/README_data.md`。

### 特征决策规则

```text
能复用标准特征且与 MLCL baseline 公平对齐 → 优先复用；
复用会造成字段缺失、口径冲突或无法支撑主模型 → 自建特征链路；
无论哪种，必须固化特征版本，不允许混用。
```

### 输出

```text
new_code/reports/P1_feature_plan.md
new_code/data/README_data.md
new_code/HANDOFF_PHASE_01.md
```

---

## 3. P2 主模型实现

### 目标

实现主模型最小可运行版本：sLSTM + AWAF + 多任务预测头。

### 模块

```text
models/encoders/slstm.py
models/fusion/awaf.py
models/heads.py
models/ours_xlstm_fusion.py
utils/metrics.py
engine/trainer.py（最小版）
```

### 必做子任务

1. 审计已有 xLSTM 代码后，确定复用或重写；
2. 实现纯 PyTorch `SLSTMCell` 和 `SLSTMEncoder`，含指数门、normalizer、stabilizer 和 mask；
3. 实现 AWAF：
   - 跨模态上下文增强；
   - 二阶 Hadamard 交互打分；
   - 可学习温度 τ；
   - modality dropout 开关；
   - 权重缓存与落盘；
4. 实现 baseline fusion 变体：`mean / concat / gated / fixed / awaf_no_context / awaf_no_interaction`；
5. 做随机张量 forward test；
6. 做真实数据 1 epoch smoke test。

### 输出

```text
new_code/reports/P2_main_model_smoke.md
new_code/HANDOFF_PHASE_02.md
outputs/P2_smoke/...
```

### 验收

1. 前向不报错；
2. loss 可计算；
3. 可反传；
4. `w_t + w_a + w_v = 1`；
5. 权重可落盘；
6. 真实数据 smoke test 有日志和指标；
7. 无演示数据冒充。

---

## 4. P3 主模型小样本快速调参

### 目标

在真实数据子集上快速搜索关键超参，固定全量训练配置。

### 搜索范围

```text
hidden_dim: 128 / 256 / 384
num_layers: 1 / 2
dropout: 0.1 / 0.2 / 0.3
lr: 1e-4 / 5e-5 / 3e-5
batch_size: 16 / 32
pooling: masked_mean / last_valid
alpha_cls: 0.5 / 1.0
beta_aux: 0 / 0.1
use_modality_dropout: false / true
tau_init: 1.0
```

### 输出

```text
new_code/reports/P3_hparam_search.csv
new_code/reports/P3_hparam_search.md
new_code/configs/models/ours_xlstm_fusion.yaml
new_code/docs/HYPERPARAMS.md
new_code/HANDOFF_PHASE_03.md
```

---

## 5. P4 主模型全量正式训练

### 目标

固定 P3 超参后，使用全量 MOSI/MOSEI 正式训练。

### 要求

1. epoch 不低于 50；
2. warmup + cosine annealing；
3. gradient clipping；
4. early stopping；
5. 多 seed；
6. 自动保存 best/last pth；
7. 每个 run 更新 `experiment_registry.csv`。

### 硬性目标

```text
MOSI ACC2_Non0 ≥ 88%
MOSI F1_Non0 ≥ 88%
MOSEI 多个主要指标较强或有可解释提升
```

如未达标，必须追根因并输出问题报告，不得粉饰。

---

## 6. P5 代码重构

### 目标

在主模型跑通并完成调参后，再将项目整理成标准深度学习项目结构。

### 原则

1. 先写 `docs/FILE_STRUCTURE.md`，再动手移动/新建代码；
2. 先迁移已经跑通的主模型，确保指标不变；
3. 建立统一 `BaseModel / registry / trainer / evaluator / metrics`；
4. 支持 `train_main.py` 默认按顺序训练全部模型，且支持 `--model` 单跑；
5. 支持 `eval_main.py` 一次性评估全部模型并输出表图；
6. 必要时直接重新撰写旧文件对应功能，不做脆弱补丁；
7. 不参考旧主模型架构图。

### 输出

```text
new_code/docs/FILE_STRUCTURE.md
new_code/reports/P5_refactor.md
new_code/HANDOFF_PHASE_05.md
README.md 更新结构树和运行方法
```

---

## 7. P6 baseline 接入

### 目标

统一接入经典 baseline、MLCL、CASP。

### 规则

1. baseline 原则上不改架构；
2. 每接入一个跑通一个；
3. 先记录原论文超参和代码来源；
4. 所有结果用 `utils/metrics.py` 统一复算；
5. CASP 单独标注 TTA；
6. MLCL 必须核对是否为 Zhuang et al. TMM 2025 与 `YetZzzzzz/MLCL`。

### 输出

```text
new_code/reports/P6_baseline_status.md
new_code/HANDOFF_PHASE_06.md
outputs/P6_baselines/...
```

---

## 8. P7 消融实验

最低消融集合：

```text
Full AWAF-sLSTM-Fusion
w/o AWAF → mean
w/o AWAF → concat
w/o AWAF → gated
fixed global weights
awaf_no_context
awaf_no_interaction
sLSTM → LSTM
sLSTM → GRU
w/o aux loss
+DEConv
+Data2Vec-Audio
+CME
with / without modality dropout
```

所有消融必须同数据、同特征、同指标、同可复现记录。

---

## 9. P8 表格与图像输出

必须输出：

```text
outputs/_summary/<timestamp>/main_comparison.xlsx
outputs/_summary/<timestamp>/main_comparison.docx
outputs/_summary/<timestamp>/ablation_results.xlsx
outputs/_summary/<timestamp>/ablation_results.docx
outputs/_summary/<timestamp>/all_models_loss_compare.png
outputs/_summary/<timestamp>/awaf_weight_distribution.png
outputs/_summary/<timestamp>/awaf_weight_by_sentiment_bins.png
outputs/_summary/<timestamp>/ablation_barplot.png
outputs/_summary/<timestamp>/error_case_weights.png
```

图像要求 ≥900 dpi，不允许终端截图。表格为三线表，标明 Non0、Has0、CASP TTA、特征版本和结果可信度。

---

## 10. P9 论文重写

只有在 P4/P7/P8 固定后，才启动论文第 3/4/5 章重写建议。

输出：

```text
reports/P9_chapter3_rewrite.md
reports/P9_chapter4_5_rewrite.md
HANDOFF_PHASE_09.md
```

论文修改建议必须包含：

```text
修改位置
原问题
修改理由
建议替换内容
注意事项
```

---

## 11. P10 最终清理

只有用户明确指令后才能清理。先输出：

```text
reports/final_cleanup_plan.md
```

用户确认后再删除中间文件、过时结果、重复日志和无效缓存。删除前必须备份。

---

## 12. 异常处理

遇到错误：

1. 保存完整错误日志；
2. 给出最小复现命令；
3. 判断是路径、环境、数据、显存、代码还是依赖；
4. 尝试最小修复；
5. 记录到报告；
6. 解决不了就如实汇报，不编造成功。

---

## 13. 阶段汇报格式

```text
本阶段完成：
- ...

关键发现：
- ...

新增/修改文件：
- ...

实验结果：
- ...

无法完成或仍需确认：
- ...

下一步建议：
- ...
```


---

## 14. 外部代码下载、联网检索与依赖安装流程

当某个模块、baseline 或依赖在本地缺失时，CC/Codex 可以联网检索并下载公开资源，但必须按以下流程执行：

1. 在 handoff 或阶段报告中写明下载目的：主模型模块、候选模块、baseline、数据特征、依赖库或工具脚本。
2. 优先选择官方论文仓库、作者主页、GitHub release、官方文档；不得使用来源不明的镜像。
3. 下载到规范目录：`external/`、`third_party/`、`models/baselines/`、`data/raw_external/` 或 `baselines/<model_name>/`。
4. 建立 `docs/EXTERNAL_SOURCES.md`，记录：名称、URL、commit/release、许可证、下载时间、用途、是否修改、是否纳入实验。
5. 安装 Python 依赖时优先使用新环境 `mme_xlstm`；若依赖冲突，为 MLCL、CASP 等建立专属环境；不得污染旧 `mme`，除非用户明确同意。
6. 所有安装命令写入 `env/install_commands.md`，并导出 `environment.yml` 与 `pip freeze`。
7. 下载与安装完成后必须做 smoke test，确认依赖没有破坏主模型训练入口。

---

## 15. 文件夹清理流程：P10 才能执行，且必须用户确认

“无用文件夹”不能自动删除。允许自动扫描和生成清理计划，但清理动作必须等用户确认。

### 15.1 可自动扫描的对象

```text
旧 handoff、过时报告、重复实验目录、临时缓存、异常命名文件、空目录、无引用中间文件、大型临时特征、失败 run 产生的不完整产物
```

### 15.2 清理报告必须包含

```text
路径 | 类型 | 大小 | 产生阶段 | 是否有备份 | 是否可复现 | 建议动作（保留/归档/删除） | 风险说明
```

### 15.3 执行规则

1. P0/P5 可生成 `reports/cleanup_candidates.md`，但不得清理；
2. P10 输出 `reports/final_cleanup_plan.md`；
3. 用户确认后，优先移动到 `archives/cleanup_<timestamp>/`；
4. 只有用户明确要求删除时才真正删除；
5. 清理后追加更新 `memory.md` 与 `docs/DECISIONS.md`。



---
## P6M Baseline-Lite 路线 (2026-06-19)
Baseline 路线固定为内部 baseline-lite 轻量复现。外部仓库仅作结构参考/引文来源/原论文报告值来源。
固定 baseline: TFN-lite, LMF-lite, MulT-lite, MISA-lite, SelfMM-lite, MMIM-lite, MLCL-lite, DLF-lite。
第一批: TFN-lite, LMF-lite, MulT-lite, SelfMM-lite。
CASP 不入普通baseline主列(TTA)。DPDF-LQ/DashFusion/R3DG暂不执行。
原论文报告值标注 'Reported by original paper'。Baseline-lite结果必须来自真实训练+utils/metrics.py统一复算。
教学演示/模拟/占位不得进入论文正式实验表。
