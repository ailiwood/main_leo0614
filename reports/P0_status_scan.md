# P0 现状扫描报告

> 生成时间：2026-06-16  
> 阶段：P0 初始化扫描  
> 状态：完整扫描完成，未修改任何文件，未执行训练/下载/安装

---

## 一、已读取的配置文件

| 文件 | 已读 | 核心作用 |
|------|:--:|----------|
| claude.md | ✅ | ★ 最高执行准则，项目背景/目标/权限/红线 |
| workflow.md | ✅ | ★ P0-P10 工作流程 |
| 实验设计.md | ✅ | ★ 实验设计，baseline/数据集/环境 |
| 主模型实现规划.md | ✅ | ★ 主模型最低主干+候选模块技术依据 |
| 代码重构建议.md | ✅ | ★ 目标目录结构+代码规范 |
| 最新学术进展与代码推荐.md | ✅ | 文献与代码推荐 (xLSTM/KuDA/MLCL/CASP) |
| memory.md | ✅ | 阶段记忆，当前待办 |
| 经验总结.md | ✅ | 可复用经验+避坑规则 |
| project_instruction.md | ✅ | 网页版 AI 角色+论文重构主线 |
| CC启动提示词_两阶段版.md | ✅ | CC 两阶段启动说明 |
| 0613实验FINAL_REPORT.md | ✅ | ⚠ 旧 v9 报告，仅参考不入主表 |

无 docs/*.md 文件，无 HANDOFF*.md 文件，无 README.md（已删除）。

---

## 二、扫描过的目录

| 目录 | 内容 | 关键发现 |
|------|------|----------|
| `new_code/` (根) | 10 .md + 6 .py + .docx | 当前代码全为 V9 路线 |
| `utils_models/` | 6 个 .py 文件 | lstm_v.py 为 mLSTM, AGPL-3.0 |
| `utils_tools/` | 3 个 .py 文件 | metricsTop.py 不完备 |
| `utils_train/` | 1 个 .py | 旧训练引擎 |
| `tmdc_adapter/` | 10 个 .py + features/ | MOSI 特征已提取，MOSEI 未提取 |
| `experiments/` | v9_roberta/ + v9_roberta_5seed/ | V9 5个权重文件 |
| `data/` | mosi/ + CMU-MOSEI/ | 原始数据存在，无预提取 pkl |
| `results_20260307/` | 4 .md + 4 图像 + 2 JSON | 第一轮旧结果 |
| `创新点对应论文/` | 2 个 PDF | 参考论文 |
| `logs/` | 2 个 .log | V9 训练+特征提取日志 |

---

## 三、代码结构概览

当前 `new_code` 的实际代码结构（与目标结构差异很大）：

### 3.1 实际代码入口现状

**主训练入口**：
- `train.py` (= train_main_v6.py)：V9 模型训练，使用 RoBERTa-large 特征
  - 硬编码路径：`sys.path.insert(0, r'D:\business\pycharm\project\Tri_modal_ER')`
  - 依赖不在 `new_code` 内的文件：`mosei_roberta_dataset.py`、`model_main_v6.py`
  - 仅训练 MOSEI，不支持 MOSI
  - 仅支持单个模型，无 `--model` 参数

**特征提取入口**：
- `extract_features.py`：RoBERTa-large 特征提取
  - 硬编码路径：`CASP_PKL = r'D:\business\pycharm\project\jqxxtest\CASP-main\data\casp_mosei.pkl'`
  - 依赖外部 CASP pkl 获取 audio/vision 特征

**评估入口**：
- `ensemble_5seed.py`：后处理集成脚本，仅用于 V9 模型
- 无独立 `eval_main.py`

**其他**：
- 无 `scripts/` 目录
- 无 `configs/` 目录
- 无 `engine/` 目录
- 无 `models/` 子目录结构

### 3.2 当前模型架构 vs 目标架构

| 组件 | 当前实现 | 目标实现 |
|------|----------|----------|
| 文本编码器 | TransformerEncoder (ModalEncoder) | sLSTM 时序编码 |
| 音频编码器 | TransformerEncoder (ModalEncoder) | sLSTM 时序编码 |
| 视觉编码器 | TransformerEncoder (ModalEncoder) | sLSTM 时序编码 |
| 融合模块 | GatedFusion (softmax门控) | AWAF (两段式自适应加权) |
| 预测头 | 回归+分类 (无单模态辅助) | 回归+分类+单模态辅助头 |
| 损失函数 | class_weighted_l1 + BCE | 多任务损失 (含 aux) |

**结论：当前 V9 模型与目标主模型架构完全不兼容，必须重写。**

---

## 四、数据 / 特征现状

### 4.1 原始数据

| 数据集 | 状态 | 内容 |
|--------|------|------|
| CMU-MOSI | ✅ 存在 | label.csv + 93 视频帧目录 + 93 原始视频 + 93 音频目录 |
| CMU-MOSEI | ✅ 存在 | Audio_chunk (Train/Val/Test) + Labels CSV ×5 |

### 4.2 预提取特征

| 特征 | 位置 | 格式 | 状态 |
|------|------|------|------|
| RoBERTa-large MOSEI | 旧项目 `Tri_modal_ER/data/CMU-MOSEI/*_roberta.pkl` | (50, 1024) + (64, 80) + (64, 176) | ⚠ 在旧项目路径，不在 new_code |
| CASP MOSEI | 外部 `jqxxtest/CASP-main/data/casp_mosei.pkl` | audio/vision/regression_labels | ⚠ 在外部路径 |
| MOSI 模拟特征 | 旧项目 `Tri_modal_ER/data/mosi_simulated_features.pkl` | — | ⚠ 名含"simulated"，需审计 |
| TMDC MOSI 特征 | `tmdc_adapter/features/mosi/` | per-frame npy + combined pkl | ✅ 已提取 |
| TMDC MOSEI 特征 | `tmdc_adapter/features/mosei/` | (空目录) | ❌ 未提取 |

### 4.3 特征链路状态评估

| 模态 | MOSI | MOSEI |
|------|------|-------|
| Text | TMDC DeBERTa-large npy (已提取) | 未提取 |
| Audio | TMDC wav2vec-large npy (已提取) | 未提取 |
| Vision | TMDC MANet npy (已提取) | 未提取 |

### 4.4 关键判断

1. **当前 `new_code/data` 中没有可用的预提取三模态特征 pkl 文件**，无法直接支撑 P2 smoke test
2. V9 路线使用的 RoBERTa-large pkl 特征在旧项目路径，且 text dim=1024（与目标不一致）
3. MOSI 有"simulated"特征标记，需确认是否为真实特征还是模拟数据
4. TMDC 链路已为 MOSI 提取了特征（DeBERTa+wav2vec+MANet），MOSEI 未提取
5. **推荐 P1 优先评估复用 MLCL 标准特征**（如公开可用），以避免特征提取的工程负担并确保 baseline 公平性

---

## 五、历史模型与旧代码审计

### 5.1 lstm_v.py — NX-AI xLSTM 实现

| 审计维度 | 结论 |
|----------|------|
| **实现类型** | **mLSTM** (Matrix LSTM)，非 sLSTM |
| **是否包含 sLSTM** | ❌ 不包含。`MatrixLSTMCell` + `parallel_stabilized_simple` 是矩阵记忆 mLSTM |
| **是否包含指数门** | ❌ 使用 logsigmoid forget gate + input gate preact，非 sLSTM 指数门 |
| **是否包含 normalizer** | ⚠ 包含 `MultiHeadLayerNorm` 作为 `outnorm`，但非 sLSTM standard normalizer |
| **是否包含 stabilizer** | ✅ `parallel_stabilized_simple` 有 stabilize_rowwise |
| **是否包含 mask 更新** | ❌ 无显式 mask 更新逻辑 |
| **是否包含 NaN 防护** | ✅ eps=1e-6，+ exp stabilization |
| **许可证** | **AGPL-3.0** ⚠ |
| **来源** | NX-AI GmbH, official xLSTM repository |
| **可复用性** | ❌ **不可直接复用**。原因：(1) 是 mLSTM 非 sLSTM；(2) AGPL-3.0 不能混入主模型；(3) 接口与三模态时序编码需求不匹配 |
| **建议** | **必须重写 sLSTM**。参考试公式和 stabilizer 思路，但必须纯 PyTorch 自实现 |

### 5.2 attention_encoder.py — vision_xLSTM

| 审计维度 | 结论 |
|----------|------|
| **实现类型** | 使用 `ViLBlock` 来自 lstm_v.py → 继承 mLSTM + AGPL-3.0 |
| **可复用性** | ❌ 不可复用（依赖 AGPL-3.0 代码） |
| **建议** | 重写 |

### 5.3 DEConv.py — DEConv_2

| 审计维度 | 结论 |
|----------|------|
| **实现类型** | 动态卷积，5组卷积核加权融合 |
| **许可证** | 未标注（自写代码） |
| **可复用性** | ⚠ 部分可复用。但：(1) 硬编码 32×32 reshape；(2) 仅适配 1024 维输入；(3) 需适配为目标视觉增强模块 |
| **建议** | 作为候选模块参考实现，P2 时决定是适配复用还是重写 |

### 5.4 ours_model.py — rob_d2v_MATF

| 审计维度 | 结论 |
|----------|------|
| **架构** | RoBERTa-large + Data2Vec-Audio + DEConv + CME + selfTransformer |
| **是否含 sLSTM** | ❌ 否 |
| **是否含 AWAF** | ❌ 否 |
| **与目标架构关系** | 完全不同 |
| **建议** | 仅作为 CME/Data2Vec-Audio/DEConv 候选模块的挂载参考，不作为主模型基础 |

### 5.5 metricsTop.py — 旧指标模块

| 审计维度 | 结论 |
|----------|------|
| **输出指标** | MAE, Corr, Has0_acc_2, Has0_f1 (仅4个) |
| **缺少指标** | ❌ ACC2_Non0, F1_Non0, ACC7 |
| **关键问题** | `ACC2` 计算含 zero label (`pred_binary = (preds >= 0)`)，非 Non0 口径 |
| **建议** | **必须重写**为 `utils/metrics.py` |

---

## 六、baseline 状态

### 6.1 经典 baseline (5个核心)

| baseline | 代码存在 | 来源清楚 | 许可证 | 运行说明 | 结果/日志 | per-sample pred | 统一 metrics 复算 | 风险 |
|----------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|------|
| TFN | ❌ | — | — | — | — | — | — | 需从 MMSA 框架或官方引入 |
| LMF | ❌ | — | — | — | — | — | — | 同上 |
| MulT | ❌ | — | — | — | — | — | — | 同上 |
| MISA | ❌ | — | — | — | — | — | — | 同上 |
| Self-MM | ❌ | — | — | — | — | — | — | 同上 |

### 6.2 2025 新 baseline (2个)

| baseline | 代码存在 | 来源核对 | 结果/日志 | 风险 |
|----------|:--:|------|:--:|------|
| MLCL | ❌ | YetZzzzzz/MLCL (TMM 2025, DOI: 10.1109/TMM.2025.3613116) 待核对 | 无 | 0613 报告中的 MLCL 84.00% 基准待核实是否同源 |
| CASP | ❌ | zrguo/CASP (AAAI 2025) 待核对 | 无 | TTA 属性需单独标注 |

### 6.3 baseline 接入状态总结

**所有 7 个 baseline（5 经典 + 2 新）均不在 `new_code` 中。** 需在 P6 统一接入。

---

## 七、旧架构图与旧 handoff 状态

| 产物 | 状态 | 说明 |
|------|------|------|
| 旧主模型架构图 | **作废** | 所有配置文件一致声明，不得参考 |
| 旧 HANDOFF | **不存在** | 已删除 (memory.md 记录) |
| `0613实验FINAL_REPORT.md` | **仅参考** | v9 RoBERTa 86.81% 结果，不含 xLSTM/AWAF/DEConv/CME |
| `results_20260307/` | **仅参考** | 第一轮消融结果，指标口径未统一 |
| 论文初稿 docx | **存在** | 3MB, 2025-03-18, P0 仅做结构性审计 |

---

## 八、主要风险排序

| # | 风险 | 严重程度 | 解决阶段 |
|----|------|:--:|:--:|
| 1 | 无可用特征 pkl 文件，数据链路需从头建立 | 🔴 高 | P1 |
| 2 | lstm_v.py 是 mLSTM+AGPL-3.0，无法直接复用 | 🔴 高 | P2 |
| 3 | 当前所有代码是 V9 路线，与目标完全不兼容 | 🔴 高 | P2-P5 |
| 4 | 无统一 metrics.py，旧指标缺少 Non0 口径 | 🟡 中 | P2 |
| 5 | MOSI 旧特征标记为"simulated"，真实性存疑 | 🔴 高 | P1 |
| 6 | 所有 baseline 代码均不存在 | 🟡 中 | P6 |
| 7 | tmdc_adapter MOSEI 特征未提取 | 🟡 中 | P1 |
| 8 | 旧代码硬编码路径指向不存在的 D:\business\... | 🟡 中 | P2 |
| 9 | 0613 报告 MLCL 84.00% 基准来源待核实 | 🟢 低 | P6 |
| 10 | 论文初稿可能包含旧结论/旧指标 | 🟡 中 | P9 |

---

## 九、论文初稿 docx 结构性审计

### 9.1 文件属性

| 属性 | 值 |
|------|-----|
| 文件名 | `000论文初稿03181845.docx` |
| 大小 | 3,011,182 字节 (~2.9 MB) |
| 最后修改 | 2025-03-18 18:45 |
| 临时文件 | `~$0论文初稿03181845.docx` (Word 锁文件) |

### 9.2 风险评估（无需打开即可判断的风险）

基于配置文件的警告，该初稿可能存在以下风险：
1. 可能将 V9 RoBERTa-large 路线错误地写成 xLSTM-Fusion 主模型
2. 可能将未实现的 DEConv/Data2Vec-Audio/CME 写为已验证创新
3. 可能将普通门控融合写成 AWAF
4. 可能混用 ACC2_Non0 与 ACC2_Has0 指标口径
5. 可能包含参考旧主模型架构图的第 3 章方法描述
6. 可能在第 5 章有基于历史高分（非真实实验）的确定性结论

### 9.3 P0 审计结论

- 文件存在，可正常打开
- **不建议在 P0 深度阅读或修改论文内容**
- P9 正式重写论文时，需对照真实实验结果逐章核对和修正

---

## 十、明确声明

本阶段（P0）：
- ✅ 已读取全部 11 个 Markdown 配置文件
- ✅ 已扫描所有目录和代码文件
- ✅ 已审计 5 个历史模型/工具文件
- ✅ 已记录数据、特征、模型权重、实验产物现状
- ✅ 已标记所有风险

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
