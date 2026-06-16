# P3B 候选模块可行性审计

> 日期：2026-06-16

---

## 1. DEConv (`utils_models/DEConv.py`)

### 审计结论：需重写

| 维度 | 结论 |
|------|------|
| 许可证 | 无标注，自写代码 ✅ |
| 当前实现 | `DEConv_2(dim=32)`: 输入 [B,1024] → reshape [B,32,32,1] → 5×Conv2d → [B,1024] |
| 问题 | **硬编码 1024→32×32**。hidden_dim=256 无法匹配 |
| 是否可复用 | ❌ 不可直接复用 |
| 是否需重写 | ✅ 需重写为通用动态特征增强模块 |
| 许可证风险 | 无 ✅ |

### 重写方案

实现 `DynamicFeatureEnhancer`：
- 输入: [B, T, D] 或 [B, D]
- 内部: optional 1D conv over T + FC enhancement
- 输出: [B, T, D] 或 [B, D]
- 挂载位置: vision projection → DEConv → vision sLSTM

### T=1 边界说明

当前 vision 为 T=1 单向量，DEConv 无法做 real spatial conv。实际效果为"dynamic feature enhancement on clip-level embedding"。须在报告中明确标注。

---

## 2. CME (`utils_models/attention_encoder.py`)

### 审计结论：需从零重写

| 维度 | 结论 |
|------|------|
| 依赖 | `from utils_models.lstm_v import ViLBlock, SequenceTraversal` ⚠ |
| lstm_v.py 许可证 | **AGPL-3.0** ❌ |
| 是否可复用 | ❌ 不可复用（AGPL 依赖链） |
| 是否需重写 | ✅ 需从零重写轻量 CME |
| 许可证风险 | 必须完全避免 AGPL 代码 |

### 重写方案

实现 `LightweightCME`：
- 输入: h_t, h_a, h_v ∈ [B, D]
- 核心: cross-modal attention (每个模态 attend 其他两个)
- Multi-head attention 或 simple bilinear interaction
- 输出: enhanced h_t, h_a, h_v (同维度)
- 挂载: sLSTM 输出 → CME → AWAF
- 注意：CME 只做交互增强，AWAF 负责权重融合

---

## 3. Data2Vec-Audio (`utils_models/ours_model.py`)

### 审计结论：可参考加载方式

| 维度 | 结论 |
|------|------|
| 模型来源 | HuggingFace `facebook/data2vec-audio-large-960h` |
| HF 许可证 | Apache 2.0 ✅ |
| 当前使用 | `ours_model.py` 中 Data2VecAudioModel.from_pretrained() |
| 特征提取成本 | 需 HF 模型缓存（已在 tmdc_adapter/hf_cache 中）|
| 是否需重写 | 需新建 adapter 脚本，从 MOSI wav 提取 Data2Vec features |

### Data2Vec-Audio 特征提取评估

- MOSI 有 wav 文件 (data/mosi/wav/)
- Data2Vec-Audio 模型：1024d 输出
- HF 缓存中可能有此模型
- 提取难度：中等（需音频处理：resample 16kHz → model forward → mean pool）
- 估计耗时：约 1-2h（2199 个 wav 文件）

若提取成功，C2 = Data2Vec-Audio 特征替换 wav2vec2 音频特征。
若 HF 缓存无此模型且下载失败，C2 标记为暂缓。
