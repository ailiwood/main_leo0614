# docs/FINAL_MODEL_SPEC_DRAFT.md — DeepText-xLSTM-AWAF Residual 最终模型规格草案

> 状态：草案（未冻结） — P5C 首次实现和 seed42 初训
> 更新时间：2026-06-17
> 决策编号：D031

---

## 1. 模型名称

**中文**：基于 xLSTM 时序残差增强与自适应加权注意力的多模态情感分析模型

**英文**：DeepText-xLSTM-AWAF Residual Model for Multimodal Sentiment Analysis

**类名**：`DeepTextXLSTMAWAFResidual`

---

## 2. 模型结构

```text
输入: text [B,Tt,1024], audio [B,Ta,768], vision [B,Tv,768]

┌─────────────────────────────────────────────────────────┐
│ Text Branch (主判别分支)                                  │
│   Linear(1024→256) → LayerNorm → GELU → Dropout          │
│   → MaskedAttentionPooling → [B, 256]                     │
│   → DeepMLP(256→512→512→256, LN+GELU+Drop)              │
│   → h_text_base [B, 256]                                  │
│   → reg_text_base [B, 1], cls_text_base [B, 1]           │
│   (NO sLSTM — DeBERTa tokens are already contextualized) │
├─────────────────────────────────────────────────────────┤
│ Audio Branch (残差增强分支)                                │
│   Linear(768→256) → LN → GELU → Dropout                  │
│   → sLSTM 1-layer → [B, Ta, 256]                         │
│   → MaskedAttentionPooling → h_a [B, 256]                 │
├─────────────────────────────────────────────────────────┤
│ Vision Branch (残差增强分支)                               │
│   Linear(768→256) → LN → GELU → Dropout                  │
│   → sLSTM 1-layer → [B, Tv, 256]                         │
│   → MaskedAttentionPooling → h_v [B, 256]                 │
├─────────────────────────────────────────────────────────┤
│ Text Residual Branch (轻量投影, NO sLSTM)                  │
│   Linear(1024→256) → LN → GELU → Dropout                 │
│   → MaskedAttentionPooling → h_t_residual [B, 256]        │
│   (text_slstm_on = ablation only)                        │
├─────────────────────────────────────────────────────────┤
│ AWAF Residual Fusion                                     │
│   AWAF(h_t_residual, h_a, h_v)                           │
│   → z_residual [B, 256], awaf_weights [B, 3]             │
│   → delta_reg [B, 1], delta_cls [B, 1]                   │
├─────────────────────────────────────────────────────────┤
│ Final Prediction                                         │
│   reg = reg_text_base + δ_reg * delta_reg                 │
│   cls = cls_text_base + δ_cls * delta_cls                 │
│   δ_reg_init = δ_cls_init = 0.1 (可学习)                   │
└─────────────────────────────────────────────────────────┘
```

---

## 3. xLSTM 的位置

xLSTM (sLSTM) **仅用于**：
1. **Audio Branch**：wav2vec2 frame-level features → sLSTM 时序编码
2. **Vision Branch**：CLIP frame-level features → sLSTM 时序编码

**不用于** Text Branch — DeBERTa-large token features 已经通过 self-attention
完成了上下文化，叠加 sLSTM 会引入时序噪声（P5B 证据：80.2% → 76.2%）。

---

## 4. AWAF 的位置

AWAF 从"主融合器"变为"残差修正权重生成器"：

- **输入**：h_t_residual (text residual), h_a (audio encoding), h_v (vision encoding)
- **输出**：样本级三模态权重 [w_t, w_a, w_v] + 融合表示 z_residual
- **用途**：z_residual → delta_reg/delta_cls，决定对 text_base 的修正幅度和方向
- **可解释性**：AWAF 权重反映每个样本中 audio/vision 对 text 判断的修正贡献

---

## 5. 为什么 text 不再用 sLSTM

| 方案 | ACC2_NZ | 推理 |
|------|---------|------|
| Text + sLSTM | 76.2% | sLSTM 破坏 DeBERTa 的上下文化表征 |
| Text → DeepMLP | **80.2%** | 保留 DeBERTa 的完整语义，MLP 做任务适配 |

根因：DeBERTa-large 的每个 token 已经通过 24 层 disentangled attention
完成了充分的上下文建模。再在 token 序列上施加递归变换 (sLSTM)
不是"增强时序信息"而是"破坏已编码的语义结构"。

---

## 6. Residual Correction 公式

```text
reg_final = reg_text_base + δ_reg · delta_reg
cls_final = cls_text_base + δ_cls · delta_cls

其中:
  reg_text_base, cls_text_base = DeepMLP(AttnPool(DeBERTa_tokens))
  delta_reg, delta_cls = MLP_delta(AWAF(h_t_residual, h_a, h_v))
  δ_reg, δ_cls: 可学习标量, 初始值 0.1

Loss = L_reg + α·L_cls + β·L_sign + γ·L_delta_reg
L_delta_reg = |delta_reg|_mean   (鼓励小修正，文本主导)
```

---

## 7. 消融项

| Ablation | 含义 | 用途 |
|----------|------|------|
| `none` | 完整模型 | 主结果 |
| `no_residual` | 仅 text branch | 验证 multimodal 是否带来增益 |
| `no_audio` | 残差分支去掉 audio | 验证 audio 贡献 |
| `no_vision` | 残差分支去掉 vision | 验证 vision 贡献 |
| `no_awaf_mean_residual` | AWAF → mean pooling | 验证 AWAF vs simple mean |
| `text_slstm_on` | text residual 启用 sLSTM | **消融验证 text sLSTM 有害** |

---

## 8. 论文第 3 章修改要点

1. 不再描述"三路 sLSTM 平权编码 → AWAF 融合"的旧架构
2. 改为"文本主导的 DeepMLP + 音视频 xLSTM 残差增强 + AWAF 残差修正"
3. AWAF 公式不变，但其角色从"主融合器"改为"残差权重生成器"
4. xLSTM 描述为"用于非文本模态的时序信息提取"，而非"对所有模态的通用编码器"
5. 新增 residual correction 机制描述

---

## 9. 当前状态

- **模型未冻结**：P5C 首次实现，需 seed42 初训 + 至少 2 seed 验证
- **不能写最终结论**：需要消融实验完成后才能确认
- **MOSEI 未训练**：仅做维度兼容测试
- **下一步**：若 seed42 ACC2_NZ > 80.2% (超过 DeepMLP text-only)，继续多 seed；
  若 < 80.2%，说明 multimodal residual 当前无效，需调参

---

*关联文档：docs/主模型大修决策06171310.md, docs/DECISIONS.md (D031)*
