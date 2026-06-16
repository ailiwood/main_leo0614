# P1 特征链路裁决计划

> 生成时间：2026-06-16  
> 基于：P0 扫描结果、《实验设计.md》、《workflow.md》

---

## 一、核心问题

P0 扫描显示：**`new_code` 中没有可直接用于 P2 smoke test 的三模态特征 pkl 文件。**

需要回答的核心问题：
1. 使用哪套特征？（MLCL 标准特征 vs 自建特征 vs TMDC 特征）
2. MOSI 和 MOSEI 是否使用同一特征来源？
3. 特征版本如何固化？

---

## 二、候选方案

### 方案 A：复用 MLCL 标准特征（优先推荐）

**优点**：
- 可与 MLCL baseline 保持特征口径一致，增强公平性
- 节省特征提取时间
- 特征版本已知、split 已对齐

**需要确认**：
- MLCL 官方是否提供 MOSI/MOSEI 标准特征的公开下载
- 特征维度、序列长度、样本数、split 是否满足主模型需求
- 若 MLCL 仅提供部分模态特征，其他模态如何补充

### 方案 B：基于现有 TMDC 链路自建（备选）

**现状**：
- MOSI：TMDC 特征已提取（DeBERTa-large + wav2vec-large + MANet），但需验证与标签对齐
- MOSEI：TMDC 特征目录为空，需完整提取

**优点**：
- 特征链路可控
- TMDC 提取脚本已存在（tmdc_adapter/）

**缺点**：
- MOSEI 需从头提取（耗时）
- 与 MLCL baseline 特征口径不一致
- MOSI 旧特征（Tri_modal_ER）标记为"simulated"

### 方案 C：修复 V9 特征链路（不推荐）

利用旧 RoBERTa-large + CASP audio/vision 特征，但：
- RoBERTa-large 1024d 与目标模型的 modular design 不一致
- 旧特征在外部项目路径，不可控
- V9 路线已被明确排除为主线

---

## 三、P1 执行步骤

1. **联网检索 MLCL 标准特征**（仅检索，不下载）：
   - 访问 `https://github.com/YetZzzzzz/MLCL` 查看 README、特征下载说明
   - 记录特征维度、格式、split、样本数、许可证

2. **审计 TMDC 已提取 MOSI 特征**：
   - 读取 `tmdc_adapter/features/mosi_pkls/CMUMOSI_features_raw_2way.pkl`，确认格式和维度
   - 核对标签与官方 MOSI label.csv 的一致性
   - 判断是否为"simulated"还是真实提取

3. **评估 MOSEI TMDC 提取成本**：
   - 确认 tmdc_adapter 提取脚本是否可直接运行
   - 估算提取时间和显存需求

4. **输出决策**：
   - `data/README_data.md`：固化特征版本说明
   - `reports/P1_feature_plan.md`：裁决结果和理由

---

## 四、决策标准

```text
能复用 MLCL 标准特征且与 baseline 公平对齐 → 优先复用方案 A
MLCL 特征不可用或字段缺失 → 启动方案B自建
无论哪种，必须固化特征版本，不允许混用
MOSI 与 MOSEI 必须使用同一特征来源和版本
```

## 五、需用户/网页版 AI 判断

1. 是否允许 P1 联网检索 MLCL GitHub 仓库（只读，不下载）？
2. 如果 MLCL 特征可用，是否允许下载（~几GB）？
3. MOSI "simulated" 特征：允许审计但不可用于正式实验？
