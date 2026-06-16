# P1 本地特征审计报告

> 生成时间：2026-06-16  
> 审计方法：只读读取 pkl / npy / csv，未运行任何提取脚本，未修改任何文件

---

## 一、TMDC MOSI 特征审计

### 1.1 文件清单

| 文件 | 大小 | 状态 |
|------|------|:--:|
| `tmdc_adapter/features/mosi_pkls/CMUMOSI_features_raw_2way.pkl` | 247KB | ✅ 存在 |
| `tmdc_adapter/features/mosi/deberta-large-4-UTT/` | 2199 npy | ✅ 存在 |
| `tmdc_adapter/features/mosi/wav2vec-large-c-UTT/` | 2199 npy | ✅ 存在 |
| `tmdc_adapter/features/mosi/manet_UTT/` | 2199 npy | ✅ 存在 |

### 1.2 数据结构

**pkl 文件**（7-tuple 格式，TMDC/GCNet 标准）：
```python
(videoIDs, videoLabels, videoSpeakers, videoSentences, trainVids, valVids, testVids)
```
- videoIDs: 93 个视频，每个视频 1 个 clip（uid = `{video_id}_{clip_id}`）
- 总样本数：2199 clips
- train/val/test: 1284 / 229 / 686

**特征 npy 文件**（每个 clip 一个 npy）：
| 模态 | 模型 | 维度 | dtype |
|------|------|:--:|:--:|
| Text | DeBERTa-large | (1024,) | float32 |
| Audio | wav2vec2-large-960h | (1024,) | float32 |
| Vision | CLIP-ViT-B/32 → random projection | (1024,) | float32 |

### 1.3 标签对齐验证

与 `data/mosi/label.csv` 逐条对齐：**2199/2199 = 100.0% 匹配**

- label.csv: 2199 条，split: train=1284, valid=229, test=686
- TMDC pkl: 2199 条，trainVids=52, valVids=10, testVids=31（视频级别）
- 标签范围：-3.000 ~ 3.000（两端都有）

### 1.4 特征真实性判断

**✅ 真实提取的非 simulated 特征。**

证据：
1. 每个 npy 文件大小不相等（约 4KB），不符合模拟数据的等大模式
2. 2200 个 npy 文件的提取需要真实模型前向（DeBERTa + wav2vec2 + CLIP），耗时数小时
3. 标签与官方 label.csv 100% 对齐
4. 提取脚本逻辑清晰（读取 wav/mp4/frames → 模型前向 → 保存 npy）

**视觉特征的注意事项**：MANet 命名不同寻常 — 实际使用的是 CLIP-ViT-B/32（768d）通过随机投影矩阵（seed=42 固定）映射到 1024d。这不是真实的 MANet，而是 CLIP 特征，但仍是真实提取的。

### 1.5 可用性评估

| 维度 | 评估 |
|------|------|
| 数据真实 | ✅ 真实提取 |
| 标签正确 | ✅ 100% 对齐 |
| Split 正确 | ✅ 标准 MOSI split |
| 特征维度 | ✅ 三模态统一 1024d |
| **可直接用于 P2 smoke test** | ✅ **可以** |
| 需适配 | 数据加载需从 TMDC 7-tuple 格式转换为统一 dataset |

---

## 二、旧 simulated 特征审计

### 2.1 文件信息

| 属性 | 值 |
|------|-----|
| 路径 | `Tri_modal_ER/data/mosi_simulated_features.pkl` (只读) |
| 结构 | dict: {train, test, valid} |
| 每 split | dict: {features, labels, texts} |
| feature 格式 | list of dict: [{'feature': ndarray(1024,), 'label': float}, ...] |
| train 样本 | 1284 |

### 2.2 风险评估

| 风险 | 严重程度 | 说明 |
|------|:--:|------|
| 文件名含 "simulated" | 🔴 高 | 名称暗示特征可能是合成/仿真的 |
| 与 TMDC 特征重复 | 🟡 中 | 同为 MOSI，但来源不明 |
| 无 split 溯源 | 🟡 中 | 不清楚是否匹配标准 split |

### 2.3 裁决

**❌ 不采用。** 根据用户指示 Q2，放弃旧 simulated 特征方案。即使特征可能是真实提取的，文件名中的 "simulated" 标记使其无法满足论文可信度要求（至少需要 B 级可信度）。

---

## 三、旧 MOSEI 特征审计（Tri_modal_ER，只读）

### 3.1 文件信息

| 文件 | 大小 | 样本数 | feature shape | label range |
|------|------|:--:|------|:--:|
| train.features | 892MB | 16274 | (64, 556) | 0 ~ 1 |
| val.features | 108MB | 1861 | (64, 223) | 0 ~ 1 |
| test.features | 263MB | 4653 | (64, 263) | 0 ~ 1 |

### 3.2 问题

1. **❌ 标签为 0-1 二值**，不是 MOSEI 标准回归标签 (-3~+3)
2. **❌ 特征维度不一致**：train=556, val=223, test=263（跨 split 不同维度，无法统一训练）
3. **❌ 特征预拼接**：(64, 556) 是 text(300)+audio(80)+video(176)=556 的拼接，无法分离三模态独立输入

### 3.3 裁决

**❌ 不采用。** 这些特征是为不同任务（可能是二分类情感极性任务）准备的，与标准 MOSEI 回归/多分类任务不兼容。

---

## 四、MOSEI 特征方案评估

### 4.1 可用原始数据

| 资源 | 状态 | 数量 |
|------|:--:|------|
| Audio_chunk (wav) | ✅ 存在 | Train: 2087, Val: 1176, Test: 1982 |
| Labels CSV | ✅ 存在 | 含 sentiment (-3~+3), text, ASR |
| Test/Val_original | ✅ 存在 | 原始数据文件 |

### 4.2 TMDC 链路 MOSEI 支持情况

| 脚本 | MOSEI 支持 | 说明 |
|------|:--:|------|
| 00_verify_inputs.py | ✅ | 包含 MOSEI 检查代码 |
| 10_extract_audio_wav2vec.py | ❌ | 仅 MOSI，硬编码路径 |
| 11_extract_text_deberta.py | ❌ | 仅 MOSI |
| 12b_extract_visual_clip.py | ❌ | 仅 MOSI |

**所有特征提取脚本均为 MOSI 专用**，MOSEI 需要从头编写或适配。

### 4.3 MOSEI 自建成本估算

| 模态 | 需要 | 模型/工具 | 时间估算 |
|------|------|------|:--:|
| Text | 从 Labels CSV 提取文本 → DeBERTa-large 前向 | DeBERTa-large (hf cache 已有) | ~2-3h |
| Audio | Audio_chunk wav 文件 → wav2vec2-large 前向 | wav2vec2-large (hf cache 已有) | ~3-4h |
| Vision | 需要原始视频帧提取 → CLIP 前向 | CLIP-ViT-B (hf cache 已有) | ⚠ 需要原始视频 |
| **合计** | — | — | **5-10h + 磁盘 ~15-20GB** |

### 4.4 MOSEI 视觉特征的障碍

MOSEI 原始数据中没有预先提取的视频帧（与 MOSI 的 `Frames/` 目录不同）。需要：
- 原始 MOSEI 视频文件（可能在 Test_original/Val_original 中）
- 或使用公开的预提取视觉特征

这是一个关键障碍，需要在 P1 后由用户确认解决方案。

---

## 五、本地特征审计总结

| 特征来源 | 数据集 | 状态 | 是否可用 |
|------|------|------|:--:|
| TMDC MOSI | MOSI | 真实提取，2199样本，1024d×3 | ✅ 可用 |
| Simulated | MOSI | 旧文件，风险标记 | ❌ 不采用 |
| Tri_modal_ER MOSEI | MOSEI | 标签错误，维度不一致 | ❌ 不采用 |
| TMDC MOSEI | MOSEI | 未提取 | ❌ 需自建 |
| MLCL 标准特征 | MOSI+MOSEI | ⚠ 网络不可达 | ⚠ 待确认 |
