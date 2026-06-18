# P6I A/V Signal Quality Analysis

## 1. A/V 信号质量

| Modality | best_val_ACC2 | Signal Level |
|---|---|---|
| Audio (COVAREP 74d → 768d frozen) | 56.94% | 🔴 Very Weak (barely > random 50%) |
| Vision (OpenFace2 → 768d frozen) | 65.28% | 🟡 Moderate |
| AV (AWAF fused) | 69.44% | 🟡 Moderate |

## 2. Text + Residual 增益

| Mode | val_ACC2 | vs text_only |
|---|---|---|
| text_only | 86.57% | baseline |
| + audio residual | 87.50% | +0.93% |
| + vision residual | 87.04% | +0.47% |
| + AV residual | 87.50% | +0.93% |
| + text_confidence | 87.50% | +0.93% |

## 3. 判断

1. **Audio 有微弱信号** (56.94%)，略高于随机
2. **Vision 有中等信号** (65.28%)
3. **AV 组合更好** (69.44%)
4. **哪个补充 text**: Audio 提供最大增量 (+0.93% val)，Vision 最小 (+0.47%)
5. **是否值得继续**: 微弱。Val 增量未转化为 test gain。A/V 信号不足以产生可靠的残差增益。

## 4. Text-Confidence Residual 进展

架构修复成功：
- gate 从 0.33 → 0.75（打开）
- delta 从 0.02 → 0.23（显著增大）
- 但 residual_gain 仍为 0.00%（test）

问题从"gate 关闭"转变为"delta 方向不准确"。
