# MOSEI Canonical Text-Audio AWAF+sLSTM 模型说明

**架构已冻结，公平消融进行中** | Commit: `8a42a46`

## 1. 任务定义

输入文本序列 + 音频特征序列 → 输出情感回归值 [-3, +3]。

## 2. 输入特征

| 模态 | 特征 | 维度 | 来源 |
|------|------|------|------|
| Text | RoBERTa-large CLS | [B, 1024] | `data/processed/mosei_full/roberta_cache/` |
| Audio | COVAREP 74d | [B, 100, 74] | `data/processed/mosei_full/{split}/*.npz` |

## 3. 文本编码

```
h_t = RoBERTa(x_t)  →  CLS token  →  [B, 1024]
h_t = TextMLP(h_t)  →  [B, 256]
```

Source: `models/textft_lora_xlstm_awaf_residual.py` L304-311 (`_compute_text`)

RoBERTa-large 通过 LoRA (r=16, alpha=32) 微调。仅 query/value 矩阵可训练。

## 4. 音频编码

```
X_a = COVAREP 74d  →  [B, 100, 74]
H_a' = AudioProj(X_a)  →  Linear(74→256) + LN + GELU  →  [B, 100, 256]
H_a = sLSTM(H_a', mask_a)  →  1-layer sLSTM  →  [B, 100, 256]
h_a = Pool(H_a, mask_a)  →  MaskedAttentionPooling  →  [B, 256]
```

Source: `models/textft_lora_xlstm_awaf_residual.py` L328-338 (`_compute_audio`)

sLSTM: `models/encoders/slstm.py` — exponential gating with stabilizer states.

## 5. AWAF 融合

两段式自适应加权注意力融合 (`models/fusion/awaf.py`):

**第一段：跨模态上下文增强**
```
c_t = CrossAttn(h_t, h_a)  // h_t attend to h_a
c_a = CrossAttn(h_a, h_t)  // h_a attend to h_t
ĥ_t = LayerNorm(h_t + c_t)
ĥ_a = LayerNorm(h_a + c_a)
```

**第二段：二阶交互打分**
```
g_ta = ĥ_t ⊙ ĥ_a       // Hadamard interaction
e = MLP([ĥ_t, ĥ_a, g_ta])  // [B, 3H] → [B, 2]
[w_t, w_a] = softmax(e / τ)
z = w_t · ĥ_t + w_a · ĥ_a    // [B, H]
```

Note: AWAF 内部创建 3 模态权重 (w_t, w_a, w_v)，但 vision 为 dummy zeros，
w_v 学习趋于 0。

## 6. 预测头

```
ŷ = CanonicalHead(z)  →  Linear(256→128) + ReLU + Linear(128→1)
```

Source: `models/textft_lora_xlstm_awaf_residual.py` L220-223

## 7. 损失函数

```
L = L1Loss(ŷ, y_label)
```

无 gate/delta/auxiliary loss。

## 8. 训练配置

| 参数 | 值 |
|------|-----|
| Epochs | 12 |
| Batch size | 8 (× grad_accum 8 = effective 64) |
| Optimizer | AdamW |
| LR (text/LoRA) | 3e-5 / 3e-6 |
| Weight decay | 0.03 |
| Early stop | patience=4, min_epochs=3 |
| Seed | 42 |

## 9. 可解释性输出

- `awaf_weights_test.csv`: per-sample [w_t, w_a, w_v]
- `predictions_test.csv`: per-sample ŷ

## 10. 参数

| 组件 | 参数量 |
|------|--------|
| RoBERTa-large | ~355M (LoRA: ~3M trainable) |
| Audio projection | ~20K |
| sLSTM | ~400K |
| AWAF | ~50K |
| Canonical head | ~33K |
| **Total trainable** | **~3.6M** |

## 11. 当前证据状态

| 证据 | 状态 |
|------|:---:|
| G1 静态验证 | ✅ PASS |
| G2 动态验证 | ✅ PASS |
| G3 过拟合测试 | ✅ PASS |
| MOSEI control (87.83%) | ✅ PASS |
| MOSI control | ❌ BLOCKED |
| Fair ablation | 🔄 4 variants running |

## 12. 不应声称的结论

1. ❌ 不可说 "MOSI 上验证了 AWAF+sLSTM"
2. ❌ 不可说 "MOSEI 音频显著提升"
3. ❌ 不可说 "AWAF 在所有数据集上均有效"
4. ❌ 当前结果标记为 "待公平消融完成后最终裁定"

Generated: 2026-06-20 | Commit: `8a42a46`
