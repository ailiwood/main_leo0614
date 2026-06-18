# P6H MOSI Near-Miss Analysis: AWAF/Gate/Delta 残差融合无效诊断

**Date**: 2026-06-18  
**Result**: final_ACC2 == text_base_ACC2 = 86.43%, residual_gain = 0.00%

---

## 1. 总判断

**text_base 健康 (86.43% > 85.37%)，但多模态残差融合完全无效。**

这不是"有效但不够强"（case C），也不是"破坏文本"（case D）。残差通路产生的修正量几乎为零，导致 final == text_base。

---

## 2. 三重根因

### 根因 1：AWAF 权重崩溃 → vision-only

```
w_t = 0.0056  (0.56%)
w_a = 0.0100  (1.00%)
w_v = 0.9844  (98.44%)
```

**原因推测**：
- Vision features 经 sLSTM 编码后方差/范数远大于 text (RoBERTa-large 1024d → MLP → 256d) 和 audio (frozen 768d → Linear → 256d → sLSTM)
- AWAF 的 MLP 输入是 `[h_t, h_a, h_v, g_ta, g_tv, g_av]`，如果 vision 的 L2 norm 远大于其他模态，softmax 会被单一模态主导
- τ_init=1.0 的温度不够大，无法软化 softmax

**修复方向**：
- 对三个模态的 pooled 向量做 LayerNorm 后再送入 AWAF
- 增加 τ 初始值（如 τ_init=3.0~5.0）使权重更均匀
- 加入 AWAF entropy regularization loss: `H(w) = -sum(w * log(w))`，鼓励权重均匀分布

### 根因 2：Delta scale 过小

```
dsr = 0.02 (init)
有效残差上限 ≈ gate × dsr × max_delta = 0.46 × 0.02 × 0.5 = 0.0046
```

回归标签范围约 [-3, +3]，残差 0.0046 几乎无影响。

**修复方向**：
- dsr 从 0.02 → 0.2（x10）
- 或让 dsr 完全可学习（去掉小的初始化约束）
- 或改用加法残差（不用乘法 gate）: `reg = text_base + delta`

### 根因 3：Gate 中等抑制

```
gate_mean = 0.4596
```

约 54% 的残差被门控抑制。

**修复方向**：
- gate bias 初始化为正（使 sigmoid 初始输出接近 0.8-0.9）
- 或降低 gate 的 dropout

---

## 3. 修复优先级

| 优先级 | 改动 | 预期效果 |
|---|---|---|
| P0 | AWAF 前加 LayerNorm(h_t, h_a, h_v) | 权重均匀化 |
| P0 | AWAF τ_init 从 1.0 → 3.0 | 权重软化 |
| P0 | dsr/dsc 从 0.02 → 0.2 | 残差可感知 |
| P1 | AWAF entropy regularization λ=0.01 | 鼓励权重分散 |
| P1 | gate bias 初始化 +2.0 (sigmoid→0.88) | 初期门控更开放 |
| P2 | max_delta 从 0.5 → 1.0 | 允许更大修正 |
| P2 | 降低 LoRA lr 保护 text_base | 避免调参中 text_base 退化 |

---

## 4. 不修复项

- text_base 86.43% 已健康，LoRA 链路正常，**不要**动 LoRA target modules / r / α
- RoBERTa 本身、tokenizer、label、split 均正常
- audio/vision feature pipeline (frozen features → sLSTM) 先不动

---

## 5. 下一轮建议

1. 应用 P0 三项修复（LayerNorm + τ + dsr）
2. 单 seed42 50ep 快速验证
3. 目标：residual_gain ≥ 0.5-1.0%（即 final ≥ 86.9-87.4%）
4. 如果 residual_gain 转正，再考虑 P1 修复并多 seed 验证
5. 在 residual_gain 稳定 > 0 前，不启动 MOSEI

---

## 6. 需用户确认

- [ ] 同意 P0 三项修复方案？
- [ ] 是否允许修改训练脚本增加 per-sample 输出保存？
- [ ] 是否需要在调参前先跑一次 text-only baseline 确认 86.43% 可复现？
