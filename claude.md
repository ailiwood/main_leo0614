# claude.md：多模态情感分析项目 CC/Codex 最高遵循文件

> 项目名称：基于扩展 LSTM 与 Transformer 的跨模态情感分析研究  
> 版本：2026-06-16 重启完善版  
> 适用对象：本地 Claude Code / Codex 执行层  
> 文件定位：本文件是 CC/Codex 在本项目中的最高执行准则。若与其他文件冲突，以本文件、用户最新明确指令和阶段 handoff 为准。  
> 总原则：第一性原理、真实代码、真实数据、真实实验、可复现、可比较、可消融、可解释。

---

## 0. 项目背景、终极目标与第一性原理

### 0.1 项目背景

本项目围绕中文硕士论文《基于扩展 LSTM 与 Transformer 的跨模态情感分析研究》展开，研究对象是文本、音频、视觉三模态情感分析，主要实验数据集为 CMU-MOSI 与 CMU-MOSEI。

上一轮项目的主要问题不是“执行不努力”，而是主线、核心创新、baseline 口径、代码结构和论文表述没有先对齐，导致以下风险：

1. 历史 v9 RoBERTa-large 路线虽然分数较高，但不包含 xLSTM、AWAF、CME、DEConv 等论文题目所需关键模块，不能直接写成“基于扩展 LSTM 与 Transformer”的论文主模型；
2. 历史 xLSTM-Fusion 路线虽然与题目更接近，但训练链路和实现细节未经重新审计，不能直接写入论文；
3. “自适应加权注意力机制”必须被实现为独立、可输出权重、可消融、可解释的 AWAF 模块，而不是普通门控融合改名；
4. baseline 曾存在同名论文混淆、指标口径混用、结果来源不可追溯等问题；
5. 旧代码结构不足以支撑一键训练、统一评估、统一指标复算、候选模块裁决和论文级表图输出。

### 0.2 终极目标

本轮项目的终极目标是建立一条完整证据链：

```text
真实主模型实现
→ 统一代码结构
→ 真实 MOSI/MOSEI 实验
→ 多 seed 主表与消融
→ AWAF 权重解释
→ 论文第 3/4/5 章按实验事实重写
```

最终论文中所有“方法描述、实验结果、创新点、结论”必须能回溯到真实代码、真实配置、真实日志、真实指标和真实产物。

### 0.3 第一性原理

本项目必须从研究问题和真实实现出发，不从旧代码、旧图、历史高分、论文初稿或上一轮失败路径倒推。

核心判断：

```text
先代码，后论文；
先真实实验，后确定结论；
先最低主干跑通，后候选模块裁决；
先统一指标口径，后比较 baseline；
先冻结最终模型，后绘制架构图和重写第 5 章。
```

### 0.4 核心任务

本轮核心任务不是简单修补旧文件，而是建立一个可运行、可复现、可消融、可解释的论文主模型与实验工程：

```text
三模态输入 text / audio / vision
→ 线性投影到共享维度 d + mask/padding 对齐
→ 三路独立 sLSTM 时序编码
→ 样本级自适应加权注意力融合 AWAF
→ 回归头 + 分类头 + 单模态辅助头
→ MOSI/MOSEI 统一评估
→ 候选增强模块与 baseline 对比
→ 论文表图和结论生成
```

其中 AWAF 是本轮核心创新模块，必须命名为 `AdaptiveWeightedAttentionFusion`，能够输出每个样本的三模态权重 `w_t, w_a, w_v`，满足 `sum(w)=1`，并支持权重保存、可视化、样本级解释和消融。

---

## 1. 分工协作设定

### 1.1 用户

用户是项目统筹者和最终决策者，负责：

1. 提供本地路径、数据、导师要求、GitHub 仓库、关键文件和阶段产物；
2. 在 CC/Codex 与网页版 AI 之间传递 handoff、错误日志、实验报告和表图；
3. 决定是否采用某条模型路线、某个候选模块、某组 baseline 结果或某种论文写法；
4. 最终确认论文提交版本。

### 1.2 网页版 AI

网页版 AI 是研究设计与论文逻辑中枢，负责：

1. 阅读 handoff 和报告，判断当前阶段与可信度；
2. 判断模型实现、实验结果、创新点和论文表述是否一致；
3. 检索并核验文献、baseline、仓库、DOI 和引用口径；
4. 给 CC/Codex 生成阶段性可执行任务；
5. 识别风险：伪精确引文、指标口径混用、把未实现模块写成创新、把 CASP 当普通 baseline、把历史高分写成新主模型等。

网页版 AI 不直接假装跑本地实验，不编造结果，不在真实实验前写第 5 章确定性结论。

### 1.3 CC/Codex

CC/Codex 是本地执行层，负责：

1. 读取文件、审计代码、扫描数据、生成报告；
2. 在用户确认阶段任务后修改、重构、重写 `new_code` 下代码；
3. 实现模型、训练、评估、保存结果、输出表图；
4. 更新 handoff、`memory.md`、`经验总结.md` 和 `docs/DECISIONS.md`；
5. 将所有实验登记到 `reports/experiment_registry.csv`；
6. 将疑问、风险和需决策事项写入 handoff，等待用户或网页版 AI 判断。

CC/Codex 不得替用户做最终方向决策，不得伪造结果，不得用演示数据冒充真实实验，不得把不可运行代码写成已复现。

---

## 2. 路径、权限与环境边界

### 2.1 固定路径

```text
项目总根路径：E:\00project_code\main_leo
代码工作根路径：E:\00project_code\main_leo\new_code
只读参考代码：E:\00project_code\main_leo\mme\Tri_modal_ER
参考文献目录：E:\00project_code\main_leo\refer
旧 Conda 环境：mme
旧环境路径：E:\Anaconda3\envs\mme
推荐新环境：E:\Anaconda3\envs\mme_xlstm
GitHub 起点：https://github.com/ailiwood/main_leo0614
```

`Tri_modal_ER` 和 `refer` 默认只读。需要复用其中逻辑时，只能复制到 `new_code`、`external/`、`third_party/` 或 `baselines/` 等可追踪目录，并在报告中注明来源。

### 2.2 阶段性权限

#### P0 / 初始化阶段权限

P0 只允许扫描、读取、审计、整理和生成报告。不得训练正式模型，不得重构代码，不得改核心模型文件，不得删除文件。

P0 允许新增或追加以下阶段记录类文件：

```text
reports/*.md
HANDOFF_PHASE_00.md
memory.md 追加阶段记录
docs/DECISIONS.md（如有关键判断）
经验总结.md（如有可复用经验）
```

#### 正式代码修改阶段权限

自 P2 起，在用户明确确认进入代码修改阶段后，CC/Codex 可以在 `new_code` 内自动执行以下操作：

1. 修改已有代码文件；
2. 重构文件结构；
3. 新建模块、配置、脚本和测试；
4. 必要时直接重写旧代码；
5. 下载公开 baseline 代码、公开模型适配代码或公开特征下载脚本；
6. 在新建环境或 baseline 专属环境中安装 Python 包；
7. 生成训练、评估、表图、日志和实验登记产物。

但必须遵守：

1. 修改前记录涉及文件和原因；
2. 涉及覆盖或大规模移动时先备份；
3. 不修改只读参考路径；
4. 不自动删除文件夹；
5. 不执行来源不明脚本；
6. 不把外部代码冒充自研创新；
7. 所有依赖安装和外部代码下载必须记录。

### 2.3 环境管理

推荐新建主环境：

```text
mme_xlstm
```

如果 MLCL、CASP、经典 baseline 与主模型依赖冲突，允许建立专属环境：

```text
mme_mlcl
mme_casp
mme_baselines
```

所有环境操作必须记录到：

```text
env/install_commands.md
env/environment_mme_xlstm.yml
env/pip_freeze_mme_xlstm.txt
env/requirements_main.txt
env/requirements_mlcl.txt
env/requirements_casp.txt
```

---

## 3. 必须首先阅读的文件

每次新会话或项目重启，CC/Codex 必须先读取代码工作根路径下全部 `.md` 文件，再做任何计划。

必须递归扫描：

```bat
cd /d E:\00project_code\main_leo\new_code
dir /s /b *.md
```

至少应覆盖：

```text
claude.md
workflow.md
实验设计.md
主模型实现规划.md
代码重构建议.md
最新学术进展与代码推荐.md
memory.md
经验总结.md
README.md（如存在）
HANDOFF*.md（如存在，只记录，不盲从）
reports/*.md（如存在，只记录，不盲从）
docs/*.md（如存在）
```

如果 `E:\00project_code\main_leo` 与 `new_code` 下存在同名配置文件，必须在报告中记录版本冲突，并优先采用用户最新上传或 `new_code` 下最新明确版本，不能凭记忆执行。

---

## 4. 总体工作顺序链路

项目主线阶段如下：

```text
P0 项目理解与现状扫描
→ P1 数据与特征链路裁决
→ P2 最低主干实现：sLSTM + AWAF + 多任务头
→ P3 小样本调参与候选模块初筛
→ P4 MOSI/MOSEI 全量正式训练
→ P5 工程重构与统一入口
→ P6 baseline 接入与统一评估
→ P7 正式消融与权重解释
→ P8 论文级表格、图像、xlsx/docx 产物
→ P9 按实验事实重写论文第 3/4/5 章
→ P10 用户确认后的归档与清理
```

核心顺序不可颠倒：

1. 先确认数据与特征；
2. 再实现最低主干；
3. 再小样本调参；
4. 再全量训练；
5. 再接入 baseline；
6. 再做正式消融；
7. 最后重写论文。

论文第 3 章方法描述必须由真实代码反推；第 4/5 章实验设计和结果分析必须由真实实验支撑；摘要、结论和创新点必须等最终模型冻结后再写。

---

## 5. 主模型设计与候选模块原则

### 5.1 最低必须实现主干

```text
输入：text / audio / vision 三模态序列特征
  ↓
统一投影：Linear(d_m → d) + LayerNorm + Dropout
  ↓
三路 sLSTM 时序编码：H_t, H_a, H_v
  ↓
Masked pooling 得到 h_t, h_a, h_v
  ↓
AWAF 样本级自适应加权注意力融合
  ↓
多任务预测头：回归 + 分类 + 单模态辅助头
```

最低主干是所有候选模块实验的共同基线，不等于最终模型一定只包含这些模块。

### 5.2 sLSTM 实现原则

1. 默认采用纯 PyTorch 自实现，避免官方 xLSTM 的 triton/CUDA kernel 在 Windows + conda 环境下带来不确定性；
2. 实现前必须审计历史 `utils_models/lstm_v.py`、`vision_xLSTM` 等文件；
3. 必须判断旧实现是 sLSTM、mLSTM 还是普通 LSTM 变体；
4. 必须确认是否包含指数门、normalizer、stabilizer、mask 更新和 NaN 防护；
5. 如果公式不清、来源不清、许可证不清或接口不匹配，应直接重写。

### 5.3 AWAF 实现原则

AWAF 必须独立实现为：

```python
AdaptiveWeightedAttentionFusion
```

核心逻辑：

```text
第一段：轻量跨模态上下文增强
第二段：一阶模态摘要 + 二阶 Hadamard 交互项生成样本级权重
```

权重生成形式：

```text
g_ta = h_t ⊙ h_a
g_tv = h_t ⊙ h_v
g_av = h_a ⊙ h_v

e = MLP([h_t, h_a, h_v, g_ta, g_tv, g_av]) ∈ R^3
w = softmax(e / τ)
Z = w_t h_t + w_a h_a + w_v h_v
```

AWAF 必须支持以下消融：

```text
awaf
awaf_no_context
awaf_no_interaction
mean
concat
gated
fixed
modality_dropout_off
```

### 5.4 候选增强模块

DEConv、Data2Vec-Audio、CME 不是默认关闭后忽略，而是候选增强模块。是否进入最终主模型必须由真实实验裁决。

| 模块 | 定位 | 默认状态 | 进入最终模型条件 |
|---|---|---|---|
| DEConv | 视觉动态增强候选模块 | 候选 | 稳定提升且成本可接受 |
| Data2Vec-Audio | 音频高层表征增强候选模块 | 候选 | 稳定提升且特征链路可复现 |
| CME | 显式跨模态交互候选模块 | 候选 | 稳定提升且能解释与 AWAF 的互补性 |

最终模型冻结标准：

1. ACC2_Non0 或 F1_Non0 稳定提升；
2. MAE / Corr 不明显退化；
3. 至少 2 个 seed 趋势一致；
4. 参数量、训练时间、显存成本可接受；
5. 能形成论文中可信解释。

---

## 6. baseline 体系与文献口径

### 6.1 baseline 分层原则

本项目 baseline 分为两层：

```text
经典基线：TFN, LMF, MulT, MISA, Self-MM
2025 新基线：MLCL, CASP
```

经典基线用于说明本文方法相较于传统张量融合、低秩融合、Transformer 跨模态注意力、表示解耦和自监督多任务学习的改进；2025 新基线用于增强论文时效性，但必须严格区分方法性质和比较口径。

所有 baseline 结果优先通过统一 `utils/metrics.py` 复算。若只能使用论文报告值或外部日志，必须标注可信度和限制，不能直接混入主表。

### 6.2 经典 baseline

#### TFN：Tensor Fusion Network

参考：Zadeh et al., *Tensor Fusion Network for Multimodal Sentiment Analysis*, EMNLP 2017.

定位：早期高阶张量融合代表方法。TFN 通过张量外积显式建模单模态、双模态与三模态高阶交互，是多模态情感分析经典基线。

项目中用途：

1. 作为“显式高阶交互融合”的经典对照；
2. 说明本文 AWAF 的二阶交互不是简单复制 TFN，因为 AWAF 生成的是样本级三模态权重，并保留可解释权重输出；
3. 记录其主要局限：张量维度膨胀、计算成本高、对现代预训练特征和长程时序建模支持不足。

#### LMF：Low-rank Multimodal Fusion

参考：Liu et al., *Efficient Low-rank Multimodal Fusion with Modality-Specific Factors*, ACL 2018.

定位：TFN 的低秩高效近似，通过低秩张量分解降低融合计算复杂度。

项目中用途：

1. 作为“高效张量融合”的经典对照；
2. 说明本文 AWAF 不是仅做低秩压缩，而是显式输出样本级动态模态贡献；
3. 记录其局限：仍属于早期融合范式，对深层跨模态语义交互与长程时序依赖建模不足。

#### MulT：Multimodal Transformer

参考：Tsai et al., *Multimodal Transformer for Unaligned Multimodal Language Sequences*, ACL 2019.

定位：将 Transformer 引入未对齐多模态序列建模的经典方法，核心为 directional pairwise cross-modal attention。

项目中用途：

1. 作为“跨模态 Transformer 注意力”的经典对照；
2. 说明本文可选 CME 与 AWAF 的关系：CME 若被实验采纳，承担显式跨模态交互；AWAF 承担样本级权重融合与解释；
3. 记录其局限：多组跨模态 Transformer 堆叠带来较高训练和推理成本。

#### MISA：Modality-Invariant and Modality-Specific Representations

参考：Hazarika et al., *MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis*, ACM MM 2020.

定位：表示解耦代表方法，将各模态映射到模态不变空间和模态特异空间，缓解异质性。

项目中用途：

1. 作为“模态共享/特异表示学习”的经典对照；
2. 说明本文主干不直接采用 MISA 式解耦，而是通过 sLSTM 保留模态时序信息，通过 AWAF 学习动态贡献；
3. 若后续需要，可将 MISA 的思想作为论文相关工作讨论，不得未经实现写成本文模块。

#### Self-MM

参考：Yu et al., *Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis*, AAAI 2021.

定位：自监督多任务学习代表方法，通过自动生成单模态监督信号学习模态特异表示。

项目中用途：

1. 作为“自监督单模态辅助监督 / 多任务学习”的经典对照；
2. 说明本文单模态辅助头的目的不是复刻 Self-MM，而是辅助 AWAF 和主任务训练，并用于观察单模态贡献；
3. 如果采用 Self-MM 官方或 MMSA 框架结果，必须记录代码来源、特征版本、seed、指标口径。

### 6.3 2025 新 baseline

#### MLCL：Multi-Level Contrastive Learning for Multimodal Sentiment Analysis

正确引用口径：

```text
Zhuang, Y., Bai, W., Zhang, Y., Deng, J., Hu, Z., Zhang, X., & Ren, F. (2025).
Multi-Level Contrastive Learning for Multimodal Sentiment Analysis.
IEEE Transactions on Multimedia, 27, 9044–9058.
DOI: 10.1109/TMM.2025.3613116
官方代码：https://github.com/YetZzzzzz/MLCL
```

定位：2025 新 baseline，核心是多层级对比学习，包括 uni-modal、bi-modal、tri-modal 层级的对比约束。

项目中用途：

1. 作为 2025 年以来最新强基线之一；
2. P0 必须核对本地是否已有 MLCL 代码、特征和日志；
3. P1 必须评估是否复用 MLCL 提供的 MOSI/MOSEI 标准特征，以增强与 baseline 的公平性；
4. 不得与其他同名 MLCL/MCL 论文混淆，尤其不得把 Fan et al. T-AFFC 2025 的同名/近名工作误写成本项目 MLCL。

#### CASP：Bridging the Gap for Test-Time Multimodal Sentiment Analysis

正确引用口径：

```text
Guo, Z., Jin, T., Xu, W., Lin, W., & Wu, Y. (2025).
Bridging the Gap for Test-Time Multimodal Sentiment Analysis.
Proceedings of the AAAI Conference on Artificial Intelligence, 39(16), 16987–16995.
DOI: 10.1609/aaai.v39i16.33867
官方代码：https://github.com/zrguo/CASP
```

定位：CASP 是 test-time adaptation 方法，不是普通端到端训练 baseline。其核心是 Contrastive Adaptation 与 Stable Pseudo-label generation，用于处理测试时分布偏移。

项目中用途：

1. 作为 2025 年 TTA 类型新 baseline；
2. 表格中必须单独标注 `CASP (TTA)` 或 `CASP + backbone`；
3. 必须记录其 backbone、源域/目标域设定、测试时适配步骤、是否使用伪标签、是否改变推理流程；
4. 不得与 TFN/LMF/MulT/MISA/Self-MM/MLCL 这类普通端到端模型直接不加说明地混排。

### 6.4 baseline 实验规则

1. 经典 baseline 优先使用官方代码、MMSA 框架或可信复现代码；
2. 2025 baseline 优先使用官方仓库；
3. 所有 baseline 必须记录：代码来源、commit、许可证、环境、特征版本、运行命令、seed、日志路径；
4. 能拿到 per-sample prediction 的，必须用统一 `utils/metrics.py` 复算；
5. 只能拿到论文报告值的，标注为 `paper-reported`，可信度不得高于 B/C；
6. CASP 必须单独标注 TTA 属性；
7. baseline 不得被魔改成本文方法后仍称原 baseline；
8. baseline 指标不得混用 Non0 与 Has0。

---

## 7. 代码工程规范

### 7.1 代码结构原则

代码结构服务于五个目标：

```text
可运行、可复现、可比较、可消融、可解释
```

推荐目标结构：

```text
new_code/
├── configs/
├── data/
├── models/
│   ├── encoders/
│   ├── fusion/
│   ├── baselines/
│   └── ours_xlstm_fusion.py
├── engine/
├── utils/
├── scripts/
├── reports/
├── outputs/
├── external/
├── third_party/
├── baselines/
├── docs/
├── env/
└── archives/
```

### 7.2 代码质量要求

1. Python 代码遵循 PEP8；
2. 核心类和核心函数必须写 docstring；
3. 关键函数使用 type hints；
4. 关键逻辑必须中文注释，术语可保留英文；
5. 张量形状必须注释，如 `[B, T, D]`；
6. 配置集中到 `configs/`，禁止散落硬编码；
7. 路径通过 config 或命令行参数传入；
8. 随机性由 `utils/seed.py` 控制；
9. 模型通过 `engine/registry.py` 注册；
10. trainer/evaluator 不直接依赖具体模型类。

### 7.3 必要时直接重写

以下情况必须直接重写，而不是继续修补旧代码：

1. 旧代码入口不可运行；
2. 数据流与当前主模型冲突；
3. 不支持候选模块裁决；
4. 不支持统一 `train_main.py` / `eval_main.py`；
5. 指标口径混乱；
6. 注释缺失、命名混乱、硬编码严重；
7. 许可证或来源不清。

重写不是失败，而是本轮重构的正常策略。重写前记录旧文件路径与不复用原因；重写后必须 smoke test。

---

## 8. 指标口径与结果可信度

全项目唯一指标实现：

```text
new_code/utils/metrics.py
```

必须输出并区分：

```text
ACC2_Non0 / F1_Non0   # 主列，剔除 zero label
ACC2_Has0 / F1_Has0   # 含 zero label，辅助列
MAE / Corr / ACC7
```

结果可信度：

| 标记 | 含义 | 是否可进论文主表 |
|---|---|---|
| A | 真实数据、真实代码、完整日志、统一指标、可复现 | 可以 |
| B | 真实实验但日志或口径待补 | 暂缓 |
| C | 历史结果、来源复杂、无法完整复算 | 仅参考 |
| D | 模拟、演示、随机、不可追溯 | 禁止 |

历史高分、旧日志、论文初稿中的结果，除非能回溯到代码、配置、日志和统一指标复算，否则不能进入论文主表。

---

## 9. 产物保存与实验登记

每个 run 必须保存：

```text
outputs/<phase>/<dataset>/<model>/<YYYYMMDD_HHMMSS>_<seed>/
├── config.yaml
├── command.txt
├── train.log
├── metrics_epoch.csv
├── metrics_best.json
├── best_model.pth
├── last_model.pth
├── loss_curve.png
├── predictions_test.csv
└── awaf_weights_test.csv   # 仅主模型或含 AWAF 的模型
```

每个 run 必须追加：

```text
reports/experiment_registry.csv
```

字段至少包括：

```text
timestamp, phase, model, dataset, split, seed, feature_version,
lr, batch_size, epoch, best_epoch,
ACC2_Non0, F1_Non0, ACC2_Has0, F1_Has0, MAE, Corr, ACC7,
confidence_level, paper_usable, config_path, log_path, output_dir
```

---

## 10. 外部代码、公开模型与依赖安装规则

允许 CC/Codex 在必要时联网检索和下载公开资源，但必须满足以下规则：

1. 允许下载公开 GitHub/GitLab/官方项目主页中的开源代码、README、配置文件、特征下载脚本、必要的小型工具代码；
2. 外部代码优先放入 `external/`，经适配的小型模块放入 `third_party/`，baseline 工程放入 `baselines/`；
3. 每个外部仓库必须记录 URL、commit/release、许可证、下载时间、用途、是否修改；
4. 记录文件为 `docs/EXTERNAL_SOURCES.md` 或对应报告；
5. MIT/Apache/BSD 等宽松许可证可记录后适配；GPL/AGPL/未知许可证不得直接混入主模型代码，必须写入 handoff 请求判断；
6. 允许安装 Python 包，但优先安装到 `mme_xlstm` 或 baseline 专属环境；
7. 不得污染系统 Python，不得随意破坏旧 `mme` 环境；
8. 不得执行来源不明脚本；
9. 不得下载不明二进制文件；
10. 不得把外部代码包装成本文原创模块。

---

## 11. 文件夹清理规则

本项目不允许自动删除“看似无用”的文件夹或文件。

正确规则：

1. P0、P5、P10 可以自动扫描重复、过时、中间过程、异常命名、大文件缓存等目录；
2. 扫描结果写入 `reports/cleanup_candidates.md` 或 `reports/final_cleanup_plan.md`；
3. 未经用户确认，不得执行 `del`、`rmdir`、`rm -rf` 或批量删除；
4. 用户确认后，优先归档到 `archives/cleanup_<YYYYMMDD_HHMMSS>/` 或 `backups/cleanup_<YYYYMMDD_HHMMSS>/`；
5. 只有用户明确说“可以删除”时才真正删除；
6. 实验日志、权重、预测文件、指标文件、实验登记表、论文表格和图像不得清理，除非已有可验证备份且用户确认。

---

## 12. 论文红线

1. 不把 v9 RoBERTa-large 高分路线写成 xLSTM-Fusion 主模型；
2. 不把未实现、未验证、未纳入最终主模型的 DEConv/Data2Vec-Audio/CME 写成核心创新；
3. 不把普通门控融合写成 AWAF；
4. 不混用 ACC2_Non0 与 ACC2_Has0；
5. 不把 CASP 当普通端到端 baseline；
6. 不把 MLCL 与其他同名或近名论文混淆；
7. 不参考旧主模型架构图；
8. 不在最终实验完成前写摘要和第 5 章确定性指标；
9. 不无备份覆盖或删除文件；
10. 不追求“简单能跑”而牺牲科学性和可复现性；
11. 不给未经核实的卷期、页码、DOI、GitHub 地址；
12. 不用模拟数据、随机特征或 demo 结果冒充真实实验。

---

## 13. 阶段结束汇报与 handoff

每阶段结束必须输出：

```text
本阶段完成：
关键发现：
新增/修改文件：
实验运行与核心结果：
无法完成或仍需确认：
需用户/网页版 AI 判断：
下一步建议：
```

每阶段结束必须更新：

```text
HANDOFF_PHASE_XX.md
memory.md
经验总结.md（如有可复用经验）
docs/DECISIONS.md（如有关键决策）
reports/experiment_registry.csv（如有实验）
```

不得自动进入下一阶段，除非用户明确要求连续推进。
