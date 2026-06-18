# P6H-R Residual Still Failed — 结构级分析

**Date**: 2026-06-18  
**Status**: AWAF 权重修复成功，但残差通路仍完全无效

---

## 1. 双重失败

### 失败 1：text_base 退化
- P6H original: 86.43% (lr=5e-5, 50ep)
- P6H-R best: 81.86% (lr_lora=3e-6, 42ep)
- **Δ = −4.57%**

根因：LoRA lr 从 5e-5 降至 3e-6 (1/17)，epoch 从 50 降至 42。文本分支欠训练。

### 失败 2：residual_gain 仍为零
- 5 次训练 (P6H original + R1 + R2 + R3 + R4) 全部 residual_gain = 0.00%
- AWAF 权重健康 (R2 entropy=0.99) 仍无 residual 贡献
- Gate 始终被训练关闭（0.30-0.51）
- Delta 始终被训练压缩（0.005-0.05）

---

## 2. 根因分析：为什么残差始终无效？

### 2.1 Gate 缺乏正向激励
Gate 没有 loss 鼓励开启。模型发现关闭 gate 可以减少残差噪声，降低 L1 loss。gate→0 是最安全的局部最优。

**修复**: 添加 gate_open_bonus loss：`+λ_gate * (1 - mean(gate))` 或 `+λ_gate * mean(log(gate))`

### 2.2 Audio/Vision 特征质量可能不足
- Frozen 768d features → Linear(768,256) → sLSTM → AttnPool
- 这条轻量管线可能无法从 frozen 特征中提取对情感预测有用的增量信息
- 如果 audio/vision 特征本身就是噪声，任何残差都只会增加 loss

**修复**: 
- 先用 simple classifier 测 audio-only, vision-only ACC2，确认特征含信号
- 或尝试用更强的 audio encoder (Data2Vec-Audio, wav2vec 2.0)
- 或改进 visual encoder (DEConv, ViT)

### 2.3 Text 天花板效应
- text_base ~86-87%（高 lr 下）已接近 MOSI 上 RoBERTa-large 的上限
- 剩余 ~13% 可能主要是标注噪声和模态不可约不确定性
- audio/vision 残差难以在已有的高基准上产生增量

**修复**:
- 关注 per-group 分析（near_zero 组 text_base 仅 53% → residual 最有可能在这里生效）
- R2 结果显示 residual 对 near_zero 有 +1.6% 帮助，证明残差通路在不确定性高的样本上有效
- 可设计 text-confidence-gated residual：仅在文本不确定时开启残差

### 2.4 Delta Alignment Loss 设计问题
当前 loss:
```python
lr_loss = L1(reg, label)   # 主 loss
ld = SmoothL1(effective_delta, label - text_base_detach)  # delta alignment
loss = lr_loss + 0.2 * ld
```

`label - text_base` 在 text_base 已训练好后是一个很小的值。模型的最优策略是让 delta→0、gate→0 来最小化 alignment loss，同时依赖 text_base 满足主 loss。

**修复**:
- 增加 delta alignment weight (0.2 → 1.0)
- 或去掉 delta alignment loss，只依靠主 loss 驱动 residual
- 或让 text_base 不参与主 loss 梯度 (detach), 强制 residual 学习:
  ```python
  reg = text_base_detach + gate * dsr * delta  # text_base 不参与梯度
  loss = L1(reg, label)  # 只有 residual 可以优化
  ```

---

## 3. 下一轮结构救援建议

### 方案 A：强制残差学习（推荐先试）
1. Text_base 从梯度中 detach（冻结），仅 residual 可训练
2. 这迫使模型通过 gate+delta 减少 loss
3. 如果 residual 仍不学习 → 说明 audio/vision 特征无增量信息
4. 如果 residual 开始学习 → 说明门控+残差架构可行

### 方案 B：Text-confidence conditioned residual
1. 对 high text-confidence (|reg_text_base| 远离 0) 样本，关闭 residual
2. 对 low text-confidence (|reg_text_base| 近 0) 样本，开启 residual
3. 使用 UncertainyGuidedResidualGate 已有的 margin 机制

### 方案 C：Concat-style residual
1. 不用 AWAF 加权融合 + delta expert
2. 改为 `residual = MLP(concat(h_t, h_a, h_v, uncertainty_features))`
3. `reg = text_base + residual`（不用 gate 抑制）

### 方案 D：先验证 audio/vision 单模态性能
1. 训练 audio-only / vision-only classifier
2. 如果单模态 ACC2 < 60% → 特征质量不够，需升级 encoder
3. 如果单模态 ACC2 > 65% → 特征有信号，问题在融合/残差设计

---

## 4. 建议执行顺序

1. **先提升 LoRA lr** 回 5e-5（或 1e-5），确保 text_base 恢复到 86%+
2. **执行方案 D**：验证 audio/vision 单模态 ACC2
3. **执行方案 A**：detach text_base 强制 residual 学习
4. 根据结果选择方案 B 或 C
