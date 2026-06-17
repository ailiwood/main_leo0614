# P6D 阶段进展报告 (2026-06-18)

## 一、已完成

### 1. TextFT-xLSTM-AWAF 主模型实现
- `models/textft_xlstm_awaf_residual.py`: RoBERTa-large + sLSTM + AWAF (358M params)
- `data/textft_multimodal_dataset.py`: 组合 raw text + frozen AV features
- `scripts/train_textft_xlstm_awaf_residual.py`: 三阶段训练
- Smoke test: forward/backward/AWAF/no_residual 全部通过 ✅

### 2. MOSI 关键结果

| 模型 | ACC2_Non0 | MAE | Corr |
|------|-----------|-----|------|
| Frozen DeBERTa text-only (P5B) | 80.2% | 0.885 | 0.733 |
| **RoBERTa-large fine-tuned text-only (P6C)** | **85.37%** | **0.646** | **0.826** |
| P5E V2 (frozen multimodal) | 82.47% | 0.811 | 0.739 |

### 3. MOSEI 关键结果

| 模型 | ACC2_Non0 | MAE | Corr |
|------|-----------|-----|------|
| **RoBERTa-large text-only 3ep** | **88.13%** | **0.488** | **0.816** |

**MOSEI 84% 目标已达到！** ✅ (88.13% > 84%，仅 3 epochs)

### 4. 双数据集文本基线

| 数据集 | 模型 | 样本数 | ACC2_Non0 |
|--------|------|--------|-----------|
| MOSI | RoBERTa-large text-only 30ep | 1,284 | 85.37% |
| MOSEI | RoBERTa-large text-only 3ep | 16,274 | 88.13% |

---

## 二、未完成 / 需继续

### 1. MOSI TextFT 多模态训练
- 358M 参数模型 + sLSTM + AWAF 在 16GB GPU 上训练极慢
- S1 E1 仅完成 ~70%，预计完整训练需 >2 小时
- **需优化**：减少 RoBERTa 层数、使用 LoRA、或分步训练

### 2. MOSEI 多模态
- 文本基线已达到 88.13%（超过 84% 目标）
- Audio/Visual 特征需从 WAV/MP4 提取
- Visual 特征可能不可用（需原始视频文件）

---

## 三、需要用户协助的问题

### 1. MMSDK 安装彻底失败
- PyPI: `mmsdk` 不在 PyPI 上
- GitHub: `A2Zadeh/CMU-MultimodalSDK` 仓库不存在（可能已删除/更名）
- 清华镜像: SSL 证书错误
- 默认 PyPI: 无此包

**需要用户提供**: 
- 可用的 mmsdk 安装方式（whl 文件 / conda 包 / 正确 repo 地址）
- 或者确认是否放弃 SDK，直接使用 CSV 数据（当前 MOSEI 88.13% 就是通过 CSV 实现的）

### 2. GPU 显存不足
- TextFT (358M RoBERTa + sLSTM + AWAF) 在 16GB 上训练极慢
- 当前 batch_size=2, grad_accum=8，每 epoch ~5分钟
- **建议方案**:
  - A: 使用 LoRA fine-tune (减少显存占用)
  - B: 分步训练：先训 RoBERTa text-only (快)，再单独训 residual (快)
  - C: 冻结更多 RoBERTa layers

### 3. MOSEI Vision 特征
- 数据目录只有 WAV 音频，无视频/图像文件
- 如需三模态 MOSEI，需原始视频文件或预提取的 visual features

---

## 四、当前候选排行榜

| Rank | Model | Dataset | ACC2 | Status |
|------|-------|---------|------|--------|
| 1 | RoBERTa-large text-only 3ep | MOSEI | 88.13% | ✅ 超 84% |
| 2 | RoBERTa-large text-only 30ep | MOSI | 85.37% | ✅ 超 frozen |
| 3 | P5E V2 (frozen multimodal) | MOSI | 82.47% | 基线 |
| — | TextFT multimodal | MOSI | ⏳ | 训练太慢 |

---

## 五、下一步建议

1. **优先完成 MOSI TextFT**: 方案 B（分步训练）或方案 A（LoRA）
2. **MOSEI 多模态**: 确认 vision 数据是否可用
3. **P6E 正式实验**: 若 MOSI TextFT ≥86%，即可进入多 seed + 消融
