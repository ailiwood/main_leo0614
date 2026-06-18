# P6D 最终实验结果汇总 (2026-06-18)

## 一、我们的模型结果（MOSI）

| 阶段 | 模型 | 模态 | 文本骨干 | 种子 | ACC2_Non0 | MAE | Corr |
|------|------|------|----------|------|-----------|-----|------|
| P4W | AWAF-Seq (旧) | T+A+V | DeBERTa frozen | s42 | 78.8% | 0.994 | 0.645 |
| P5B | DeepMLP text-only | T | DeBERTa frozen | s42 | 80.2% | 0.885 | 0.733 |
| P5C | DeepText-xLSTM-AWAF V1 | T+A+V | DeBERTa frozen | **s42 / s2024** | **81.10% / 81.40%** | 0.8155 | 0.7487 |
| P5C | DeepText-xLSTM-AWAF V1 | T+A+V | DeBERTa frozen | **2-seed mean** | **81.25%** | 0.8056 | 0.7489 |
| **P5E** | **UGR-AWAF Residual V2** | **T+A+V** | **DeBERTa frozen** | **s42 / s2024** | **82.47% / 81.86%** | **0.8111** | **0.7385** |
| **P5E** | **UGR-AWAF Residual V2** | **T+A+V** | **DeBERTa frozen** | **2-seed mean** | **82.17%** | **0.8299** | **0.7351** |
| P6C | RoBERTa-large text-only | T | RoBERTa fine-tuned | s42 | **85.37%** | 0.6460 | 0.8256 |

### 我们的模型结果（MOSEI）

| 阶段 | 模型 | 模态 | 文本骨干 | 种子 | ACC2_Non0 | MAE | Corr |
|------|------|------|----------|------|-----------|-----|------|
| **P6D** | **RoBERTa-large text-only 3ep** | **T** | **RoBERTa fine-tuned** | **s42** | **88.13%** | **0.488** | **0.816** |

### 失败路线（已关闭）

| 路线 | ACC2 | vs Baseline | 原因 |
|------|------|-------------|------|
| P5A Text-Guided AWAF | 75.5% | -7.0% | Text-guided破坏AV独立建模 |
| P5G wav2vec2-large audio | 80.18% | -2.3% | 1024d在小数据集过拟合 |
| P6B CLIP ViT-L/14 vision | 79.12% | -3.4% | 1024d在小数据集过拟合 |

---

## 二、Baseline 模型结果（均未本地复现）

| 模型 | 来源 | ACC2_Non0 | 状态 |
|------|------|-----------|------|
| TFN | Zadeh 2017 EMNLP | 待核 | 未复现 |
| LMF | Liu 2018 ACL | 待核 | 未复现 |
| MulT | Tsai 2019 ACL | 待核 | 未复现 |
| MISA | Hazarika 2020 MM | 待核 | 未复现 |
| Self-MM | Yu 2021 AAAI | 待核 | 未复现 |
| MLCL | Zhuang 2025 TMM | 待核 | 未复现 |
| CASP (TTA) | Guo 2025 AAAI | 待核 | 未复现，TTA单独列表 |

> ⚠️ 所有 baseline 数值均为文献报告值，未经本地复算。不可直接与我们的结果混入同一主表。

---

## 三、当前最佳候选排行榜

| Rank | 模型 | 数据集 | 模态 | ACC2 | 状态 |
|------|------|--------|------|------|------|
| 1 | P5E V2 (2-seed) | MOSI | T+A+V | 82.17% | ✅ 可进入正式实验 |
| 2 | P6C RoBERTa text-only | MOSI | T | 85.37% | ⚠️ 仅文本，需补AV |
| 3 | P6D RoBERTa text-only | MOSEI | T | 88.13% | ⚠️ 仅3ep，需补全训练 |
| — | TextFT Multimodal | MOSI | T+A+V | 训练中 | ⏳ 358M参数训练慢 |

---

## 四、架构确认

**当前所有主模型均使用三模态（T+A+V）**：
- Text: DeBERTa/RoBERTa 编码，**不使用 sLSTM**
- Audio: frozen 768d → **sLSTM** → AttentionPool
- Vision: frozen 768d → **sLSTM** → AttentionPool
- Fusion: **AWAF** + Uncertainty Gate + Delta Experts
- Final: `reg = reg_text_base + gate × δ × bounded_delta`
