# P3 候选模块裁决实验计划

> 生成时间：2026-06-16  
> 基于：《主模型实现规划.md》、《实验设计.md》

---

## 一、目标

通过真实数据实验裁决 DEConv、Data2Vec-Audio、CME 是否进入最终主模型。

**核心原则**：候选模块不是默认关闭后忽略，也不是提前写成已验证创新。必须通过真实实验裁决。

---

## 二、实验矩阵（C0-C7，共8个实验单元）

| 编号 | 结构 | 优先级 | 说明 |
|:--:|------|:--:|------|
| C0 | sLSTM + AWAF（最低主干） | 必做 | 所有候选实验的基线 |
| C1 | C0 + DEConv | 必做 | 判断视觉动态增强价值 |
| C2 | C0 + Data2Vec-Audio | 必做* | 判断音频高层表征增强价值 |
| C3 | C0 + CME | 必做 | 判断显式跨模态交互价值 |
| C4 | C0 + DEConv + CME | 建议 | 视觉增强+跨模态交互协同 |
| C5 | C0 + Data2Vec-Audio + CME | 建议 | 音频增强+跨模态交互协同 |
| C6 | C0 + DEConv + Data2Vec-Audio | 资源允许 | 双模态增强 |
| C7 | C0 + DEConv + Data2Vec-Audio + CME | 资源允许 | 全增强组合 |

*C2 若 Data2Vec-Audio 特征提取成本不可控，需在 handoff 中说明跳过原因。

---

## 三、挂载位置

### DEConv（视觉动态增强）
```text
vision feature → DEConv → vision projection → vision sLSTM
```
参考实现：`utils_models/DEConv.py` (需适配)

### Data2Vec-Audio（音频高层表征）
```text
策略 A：替换现有音频特征为 Data2Vec-Audio 提取的特征
策略 B：Data2Vec-Audio 作为额外音频分支与现有特征拼接
```
参考实现：`utils_models/ours_model.py` 中的 Data2Vec-Audio 加载方式

### CME（跨模态交互）
```text
方案 A（优先）：sLSTM 输出 H_t/H_a/H_v → CME → pooling → AWAF
方案 B：projection 后 → CME → sLSTM → AWAF
```
参考实现：`utils_models/attention_encoder.py` 中的 CMELayer
注意：原 CME 依赖 AGPL-3.0 的 ViLBlock 吗？需在实现前确认。若依赖，则需重写 CME。

---

## 四、裁决标准

候选模块纳入最终主模型必须**同时满足**：

1. ACC2_Non0 或 F1_Non0 在验证集稳定提升（≥0.5% 视为有意义）
2. MAE / Corr 不明显退化（退化>2%视为不可接受）
3. 至少 2 个 seed 方向一致
4. 参数量增加 ≤20%、训练时间增加 ≤30%、显存可接受
5. 能形成明确的论文解释

---

## 五、实验配置

### 快速实验（P3 阶段）
- 使用小样本或验证集快速实验
- epoch: 10-15（快速裁决，非正式训练）
- seed: 2-3 个
- 数据集: MOSI 全量 + MOSEI 30%采样（与旧实验对齐以节省时间）

### 正式实验（P4 阶段）
- 对通过 P3 初筛的候选组合进行全量训练
- epoch ≥ 50, 多 seed, warmup + cosine annealing

---

## 六、输出产物

```
reports/P3_candidate_module_selection.md     # 裁决报告
reports/P3_candidate_results.xlsx            # 详细结果表
reports/P3_candidate_results.docx            # 三线表
configs/models/ours_final.yaml               # 最终模型配置
docs/FINAL_MODEL_SPEC.md                     # 最终模型规格
```

---

## 七、未纳入模块的处理

对未通过裁决的候选模块：
1. 保留实验记录（指标、seed、日志）
2. 在论文中说明"为什么不采用"
3. 不得写成核心创新
4. 可作为 Discussion 或 Appendix 补充内容

---

## 八、风险评估

| 风险 | 缓解措施 |
|------|----------|
| CME 可能依赖 AGPL lstm_v.py | 实现前审计 attention_encoder.py 的 CME 依赖链 |
| Data2Vec-Audio 特征提取耗时长 | 优先评估是否有预提取特征可用 |
| 16GB 显存可能不足以同时容纳所有候选模块 | C6/C7 标记为"资源允许" |
| 若候选模块均无提升 | 最终主模型=C0，在论文中解释各模块不工作的可能原因 |
