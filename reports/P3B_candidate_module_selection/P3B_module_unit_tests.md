# P3B 候选模块随机张量测试

> 日期：2026-06-16

---

## C1: DynamicFeatureEnhancer (DEConv)

| 测试 | 结果 |
|------|:--:|
| [B,D] 输入 | ✅ shape 正确 |
| [B,T,D] 输入 | ✅ shape 正确 |
| backward | ✅ 正常 |
| NaN | ✅ 无 |
| 参数量 (dim=256) | 198,657 |

## C3: LightweightCME

| 测试 | 结果 |
|------|:--:|
| h_t/h_a/h_v [B,D] 输入 | ✅ shape 正确 |
| AWAF sum(w)=1 after CME | ✅ |
| backward | ✅ 正常 |
| NaN | ✅ 无 |
| sensitivity check | ✅ 输入变化→输出变化 |
| 初始 bug | ❌ k_proj shape error → 已修复 |
| 参数量 (dim=256) | 263,936 |

## C2: Data2Vec-Audio

| 测试 | 结果 |
|------|:--:|
| 模型加载 | ✅ data2vec-audio-base 加载成功 |
| hidden_dim | 768 |
| 特征提取 | ❌ 未完成 (暂缓) |
