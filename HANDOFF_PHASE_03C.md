# HANDOFF_PHASE_03C.md

> 阶段：P3C — 候选模块补全与最终冻结前诊断  
> 日期：2026-06-16

---

## 1. C2 Data2Vec-Audio

- 模型加载成功, 特征提取失败
- Wav2Vec2Processor 与 Data2Vec 模型不兼容
- **暂缓, 不进入 P4**

## 2. CME 不稳定性诊断

- 40ep: LightweightCME still unstable (seed 42=+0.8%, seed 2024=-2.0%)
- 40ep: GatedCME worse than C0 (both seeds -0.9% to -0.8%)
- **结论: 不是 epoch 不足, 是 CME 在 T=1 条件下不适合**

## 3. C1 DEConv 40 Epoch 复核

- 40ep vs 20ep: seed 42 退化 2.0%, seed 2024 退化 2.1%
- w_t 升至 0.77-0.84 (文本坍缩)
- **结论: 40ep 下 C1 退化为劣于 C0, 不进入 P4**

## 4. 推荐 P4 主模型

**C0 (sLSTM + AWAF only)**
- hidden_dim=256, 1 layer, dropout=0.3, lr=1e-4

## 5. 推荐 P4 消融

- sLSTM→LSTM/GRU
- AWAF modes: mean/concat/gated/fixed
- modality dropout on/off
- DEConv/CME as "attempted but not adopted" in Discussion

## 6. P4 前建议改进

- AWAF entropy regularization (防止文本坍缩)
- 加强 modality dropout

## 7. 仍需确认

- T=1 特征限制 — P4 前能否获得真实序列特征？
- MOSEI 三模态 — P4 前是否需要补齐？
