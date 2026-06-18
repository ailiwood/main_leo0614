# P6I Stage Decision

**Date**: 2026-06-19

---

## 关键问答

| # | 问题 | 答案 |
|---|------|------|
| 1 | text_base 是否恢复到 86%+ | 🟡 val=86.57%, test=83.99% (接近) |
| 2 | audio-only | 56.94% — Weak |
| 3 | vision-only | 65.28% — Moderate |
| 4 | AV-only | 69.44% — Moderate |
| 5 | text+audio 是否增益 | 🟡 val +0.93% (87.50%), test 未测 |
| 6 | text+vision 是否增益 | 🟡 val +0.47% (87.04%), test 未测 |
| 7 | text+AV 是否增益 | 🟡 val +0.93% (87.50%), test 未测 |
| 8 | text-confidence residual 是否有效 | ❌ 架构改善 (gate=0.75, delta=0.23) 但 test gain=0.00% |
| 9 | final 是否超过 text_base | ❌ 否 (85.06% = 85.06%) |
| 10 | final 是否超过 87 | ❌ 否 |
| 11 | final 是否超过 88 | ❌ 否 |
| 12 | 是否需要继续优化架构 | 🟡 架构方向正确 (gate open + delta large)，但 A/V 信号偏弱限制增益 |
| 13 | 是否允许启动 MOSEI | ❌ 否 — MOSI residual_gain 仍为 0 |
| 14 | 是否仍禁止论文第 5 章结论 | ✅ 是 — 继续禁止 |

---

## P6I 核心贡献

1. **Text-Confidence Residual 架构成功解决 gate 关闭问题**
   - gate_mean 从 0.33 → 0.75 (gate_floor=0.2 有效)
   - delta_abs_mean 从 0.02 → 0.23 (delta_loss + max_delta=1.0 有效)
   - 问题从"gate 关闭"转为"delta 方向准确性"

2. **A/V 信号验证完成**
   - Audio: 56.94% (barely above random)
   - Vision: 65.28% (moderate)
   - AV: 69.44% (moderate)
   - A/V 信号存在但不足以产生可靠的 test residual gain

3. **Val-only 增益存在但未转化为 test gain**
   - Val 上 text+audio/av 均 +0.93%，但 test 上 gain=0.00%
   - MOSI val set (229 samples) 过小，val/test gap 约 2.6%

---

## 建议

1. **P6I 架构保留** (text_confidence_residual 是正确的结构方向)
2. **不启动 MOSEI** — MOSI residual_gain 未达标
3. **考虑**：
   a. 接受 text-dominant 架构，以 text_base 为主模型
   b. 增强 A/V encoder（Data2Vec-Audio, DEConv for vision）
   c. 在 MOSEI 上直接测试 text_confidence_residual（更大数据集可能有不同表现）
