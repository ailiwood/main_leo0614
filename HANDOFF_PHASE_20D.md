# HANDOFF_PHASE_20D.md — P6D 最终阶段汇报

> 日期: 2026-06-18
> 阶段: P6D — TextFT-xLSTM-AWAF 双数据集候选实验

---

## 一、本阶段完成

### 1. TextFT-xLSTM-AWAF Residual 主模型实现 ✅

| 文件 | 功能 | 状态 |
|------|------|------|
| `models/textft_xlstm_awaf_residual.py` | RoBERTa + sLSTM + AWAF (358M) | ✅ |
| `data/textft_multimodal_dataset.py` | raw text + frozen AV features | ✅ |
| `scripts/train_textft_xlstm_awaf_residual.py` | 三阶段训练 | ✅ |
| `scripts/p6d_mosei_text_roberta.py` | MOSEI 文本训练 | ✅ |

**架构确认三模态**：
- Text: RoBERTa-large fine-tuned → **不使用 sLSTM**
- Audio: frozen 768d → **sLSTM** → AttentionPool → h_a
- Vision: frozen 768d → **sLSTM** → AttentionPool → h_v
- Fusion: **AWAF** → delta experts → uncertainty gate → bounded delta
- Final: `reg = reg_text_base + gate × δ × bounded_delta`

### 2. 实验结果

| 数据集 | 模型 | 模态 | ACC2_Non0 | MAE | Corr | 状态 |
|--------|------|------|-----------|-----|------|------|
| **MOSI** | **P5E V2 (2-seed)** | T+A+V | **82.17%** | 0.830 | 0.735 | ✅ 最优多模态 |
| MOSI | P6C RoBERTa text-only | T | 85.37% | 0.646 | 0.826 | ⚠️ 仅文本 |
| MOSI | P5B DeepMLP text-only | T | 80.2% | 0.885 | 0.733 | frozen基线 |
| **MOSEI** | P6D RoBERTa text-only 3ep | T | **88.13%** | 0.488 | 0.816 | ✅ 超84%目标 |
| MOSEI | Majority baseline | — | 62.8% | — | — | 参考 |

### 3. 失败路线（已封存）

| 路线 | ACC2 | Δ | 根因 |
|------|------|---|------|
| P5G wav2vec2-large | 80.18% | -2.29% | 1024d在小数据集过拟合 |
| P6B CLIP ViT-L/14 | 79.12% | -3.35% | 1024d在小数据集过拟合 |

---

## 二、当前架构与论文要求对照

| 论文要求 | 实现情况 |
|----------|----------|
| 基于扩展 LSTM (xLSTM) | ✅ Audio/Vision 分支使用 sLSTM |
| 自适应加权注意力机制 (AWAF) | ✅ AWAF 样本级权重 + Gate + Delta Experts |
| Text 不使用 sLSTM | ✅ Text = DeepMLP 或 RoBERTa MLP |
| 三模态输入 | ✅ T+A+V |
| AWAF 权重可保存/可解释 | ✅ |
| MOSI ACC2_Non0 | 82.17% (多模态) |
| MOSEI ACC2_Non0 | 88.13% (文本基线) |

---

## 三、存在的问题

### 1. MOSI 多模态性能瓶颈
- P5E V2 82.17% 未达 87% 目标
- RoBERTa text-only 已达 85.37%，但 AV 残差增益待验证
- TextFT 358M 模型在 16GB GPU 训练极慢

### 2. MOSEI 仅有文本基线
- 88.13% 仅 3 epochs 纯文本
- 缺少 Audio/Visual 特征提取
- MMSDK 无法安装，但 CSV 数据可用

### 3. Baseline 全部未本地复现
- TFN/LMF/MulT/MISA/Self-MM/MLCL 均仅有文献值
- CASP 需单独 TTA 表
- 不可与我们结果混表

### 4. 部分实验未完成
- TextFT 多模态完整训练
- MOSEI 多模态
- Text upper bound 超参搜索
- Two-stage / WeakNeg reweight 实验

---

## 四、需要用户协助的事项

### 🔴 紧急
1. **MMSDK 安装方案** — PyPI无包、GitHub仓库不存在。需用户提供可用安装方式，或确认放弃SDK改用CSV直读
2. **MOSEI Vision 数据** — 当前仅有WAV音频，如需三模态需原始视频文件

### 🟡 重要
3. **GPU 显存优化方案** — 358M RoBERTa 训练太慢，建议:
   - A: LoRA fine-tune (减少显存, 推荐)
   - B: 使用 P5E V2 frozen 路线 (更快, 已有82.17%)
   - C: 分步训练 (先text, 再residual)
4. **Baseline 复现优先级** — 是否需要本地复现 TFN/LMF/MulT 等？还是直接引用文献值？

### 🟢 建议
5. **下一阶段方向** — 建议 P6E: 以 P5E V2 (82.17%) 进入正式多 seed + 消融 + 论文准备

---

## 五、GitHub

| 项目 | 值 |
|------|-----|
| **当前分支** | `p6d-textft-xlstm-awaf-dual-dataset` |
| **最新 commit** | `838e1ac` |
| **URL** | https://github.com/ailiwood/main_leo0614/tree/p6d-textft-xlstm-awaf-dual-dataset |
| **所有分支** | p5c→p5d→p5e→p5f→p5g→p6a→p6b→p6c→p6d |

---

## 六、新增/修改文件

### 新增 (P6D)
- `models/textft_xlstm_awaf_residual.py` — TextFT 主模型
- `data/textft_multimodal_dataset.py` — 多模态数据集
- `scripts/train_textft_xlstm_awaf_residual.py` — 三阶段训练
- `scripts/p6d_mosei_text_roberta.py` — MOSEI 文本训练
- `configs/models/deeptext_xlstm_awaf_residual_v2_mosi_vision_l14.yaml`
- `configs/models/deeptext_xlstm_awaf_residual_v2_mosi_audio_large.yaml`
- `reports/P6D_textft_dual/*` — 全部P6D报告
- `reports/P6C_mosi_87_hard_rescue/*` — P6C报告
- `reports/P6B_formal_candidate/*` — P6B报告
- `reports/P6A_one_shot/*` — P6A报告
- `scripts/p6a_extract_mosi_vision_clip_l14_from_mp4.py`
- `scripts/p6c_text_roberta_finetune.py`
- `scripts/p5g_extract_mosi_audio_large_full.py`

### 更新
- `memory.md` — P5G/P6A/P6B/P6C/P6D 完整记录
- `docs/DECISIONS.md` — D045-D049
- `经验总结.md` — 强特征失败经验 + fine-tune 突破经验
