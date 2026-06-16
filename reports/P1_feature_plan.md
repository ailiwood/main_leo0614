# P1 特征方案裁决与特征链路规划

> 生成时间：2026-06-16  
> 基于：MLCL/CASP 网络核验、TMDC 本地审计、Tri_modal_ER 只读审计

---

## 一、三个方案的完整比较

### 方案 A：复用 MLCL 标准特征

| 维度 | 评估 |
|------|------|
| 与 MLCL baseline 公平性 | ✅ 最佳（同特征口径） |
| 特征可用性 | ⚠ 网络限制导致无法确认 |
| 特征真实性 | ⚠ 待确认（需用户浏览器访问仓库） |
| MOSEI 覆盖 | ⚠ 待确认 |
| 下载成本 | ≈ 几 GB（Google Drive / 网盘） |
| **当前可行性** | ⚠ 需用户协助确认 |

### 方案 B：基于 TMDC 链路自建（推荐）

| 维度 | 评估 |
|------|------|
| MOSI 特征 | ✅ 已提取，2199样本，1024d×3，100%标签对齐 |
| MOSEI 特征 | ⚠ 需自建，Text+Audio 可提取 (~5-7h)，Vision 需要原始视频或替代方案 |
| 特征可靠性 | ✅ 真实提取，可复现 |
| 与基线公平性 | ⚠ 与 MLCL/CASP 原始特征口径可能不一致 |
| 工程成本 | 中（MOSEI 需编写提取脚本） |

### 方案 C：旧特征链路

| 维度 | 评估 |
|------|------|
| Simulated MOSI | ❌ 已放弃（文件名风险 + P0 决定） |
| V9 RoBERTa | ❌ 已放弃（路线排除） |
| Tri_modal_ER MOSEI | ❌ 标签错误(0-1)、维度不一致 |
| CASP pkl | ❌ 在外部路径 `D:\business\...\jqxxtest\` |

**方案 C 全部不可用。**

---

##二、推荐方案

### 主推荐：方案 B（TMDC 自建） + 方案 A（MLCL）作为补充

```text
P2 smoke test (MOSI):  直接使用 TMDC MOSI 已提取特征
P4 正式训练 (MOSI):    使用 TMDC MOSI 特征
P4 正式训练 (MOSEI):   自建 MOSEI 特征链路（Text+Audio 可立即启动）
P6 baseline (MLCL):    如果能获取 MLCL 特征，用 MLCL 特征训练 MLCL baseline
                       用 TMDC 特征训练主模型，在论文中标注特征版本差异
```

### 备用方案：纯方案 A

如果用户能确认 MLCL 仓库提供完整的 MOSI+MOSEI 特征下载，则全部使用 MLCL 标准特征。
这样做的代价是 TMDC MOSI 提取工作被浪费，但与 MLCL baseline 完全公平对齐。

### 不采用方案

所有方案 C 的变体 — 理由已在第 3 节详述。

---

## 三、MOSI 与 MOSEI 特征来源统一性

```text
MOSI:  TMDC 链路（DeBERTa-large-1024d + wav2vec2-large-1024d + CLIP-ViT→1024d）
MOSEI: 同模态、同模型、同维度的 TMDC 链路自建
→ 两数据集使用统一的特征提取 pipeline，保证可比性
```

---

## 四、特征版本固化方案

### 4.1 版本标识

```text
Feature Version: TMDC-v1
Text Encoder:   DeBERTa-large (microsoft/deberta-large)
Audio Encoder:  wav2vec2-large-960h (facebook/wav2vec2-large-960h)
Vision Encoder: CLIP-ViT-B/32 (openai/clip-vit-base-patch32)
Feature Dim:    1024 (三模态统一)
MOSI:          2200 clips, 93 videos, train/val/test=1284/229/686
MOSEI:         TBD (待提取后固化)
Date:          2026-06-16 P1 decision
```

### 4.2 固化文件

```
data/README_data.md        # 特征版本说明（P1 产物）
data/mosi/                 # MOSI TMDC 特征（软链接或记录来源路径）
data/mosei/                # MOSEI TMDC 特征（P2 前自建完成后填入）
```

---

## 五、P2 smoke test 可行性

| 数据集 | 是否可立即支撑 P2 smoke test | 方案 |
|------|:--:|------|
| MOSI | ✅ 可以 | 使用 TMDC MOSI 已提取特征 |
| MOSEI | ❌ 不能 | 特征未提取 |

**短期策略**：
- P2 smoke test 先用 MOSI 数据跑通（TMDC 特征就绪）
- MOSEI 数据链路在 P2 smoke test 期间并行建设
- MOSEI P4 正式训练前必须完成特征提取

---

## 六、需要用户授权/确认的事项

1. **MLCL 仓库确认**：请用户通过浏览器访问 https://github.com/YetZzzzzz/MLCL ，确认：
   - 是否有 MOSI/MOSEI 特征下载链接（Google Drive / 百度网盘等）
   - 特征格式和维度
   - 如果可以下载，是否授权 P1/P2 下载到 `external/MLCL_features/`

2. **MOSEI 视觉特征来源**：当前 `new_code/data/CMU-MOSEI` 没有提取好的视频帧。需要：
   - 选项1：用户提供 MOSEI 原始视频文件路径
   - 选项2：使用公开的 MOSEI 视觉特征（如 FACET/OpenFace 提取结果）
   - 选项3：仅使用 Text+Audio 两模态做 MOSEI（临时方案，P4 前补齐 Vision）

3. **MOSEI 特征提取授权**：P2 期间是否允许运行 TMDC MOSEI 特征提取脚本（下载模型缓存到 hf_cache 已有，仅运行 CPU/GPU 前向）？
