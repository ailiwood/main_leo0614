# P1 MLCL/CASP 网络核验报告

> 生成时间：2026-06-16  
> 核验方式：WebSearch + WebFetch  
> 限制：GitHub 直接抓取被网络策略阻止，通过搜索引擎和缓存内容获取信息

---

## 一、MLCL：Multi-Level Contrastive Learning

### 1.1 仓库信息

| 项目 | 内容 |
|------|------|
| URL | https://github.com/YetZzzzzz/MLCL |
| 论文 | Zhuang, Y., Bai, W., Zhang, Y., Deng, J., Hu, Z., Zhang, X., & Ren, F. (2025) |
| 发表 | IEEE Transactions on Multimedia, 27, 9044–9058 |
| DOI | 10.1109/TMM.2025.3613116 |
| 框架 | UMCL (单模态) + BMCL (双模态) + TMCL (三模态) 三层对比学习 |
| 仓库可访问性 | ⚠ URL 存在于 CLAUDE.md 配置，但网络限制阻止直接抓取 README 内容 |

### 1.2 特征相关信息（从搜索引擎获取）

- **论文声称新的 SOTA** 表现于 MOSI 和 MOSEI 数据集
- 搜索未找到公开的特征下载链接或数据格式说明
- MMSA（多模态情感分析）社区通常使用标准特征集（如 MMSA 框架提供的特征）

### 1.3 核验结论

| 维度 | 结论 |
|------|------|
| 仓库存在 | ✅ 确认（URL 来自 CLAUDE.md 锁定引用） |
| 许可证 | ⚠ 无法从网络确认 |
| 特征下载 | ⚠ 无法从网络确认是否有公开下载 |
| 数据格式 | ⚠ 无法从网络确认 |
| Python/PyTorch 版本 | ⚠ 无法从网络确认 |
| **是否可直接复用** | ⚠ 待用户通过浏览器确认仓库内容 |

### 1.4 搜索遭遇的限制

WebSearch 对 MLCL 仓库内容的索引极不完整，原因可能包括：
- 仓库较新（2025年），尚未被搜索引擎充分索引
- README 可能引用 Google Drive 大文件链接
- 仓库可能在中文网络环境中受限

**建议**：由用户通过浏览器直接访问 https://github.com/YetZzzzzz/MLCL 确认 README 内容，特别关注：
1. 是否有 `data/` 目录说明
2. 是否有 Google Drive/Baidu 网盘特征下载链接
3. requirements.txt 中的依赖版本
4. 数据集 loader 代码的特征格式

---

## 二、CASP：Bridging the Gap for Test-Time MSA

### 2.1 仓库信息

| 项目 | 内容 |
|------|------|
| URL | https://github.com/zrguo/CASP |
| 论文 | Guo, Z., Jin, T., Xu, W., Lin, W., & Wu, Y. (2025) |
| 发表 | AAAI 2025 |
| DOI | 10.1609/aaai.v39i16.33867 |
| 方法 | Contrastive Adaptation + Stable Pseudo-label (TTA) |
| 许可证 | ⚠ 未从网络确认 |

### 2.2 Backbone 架构

| Backbone | 说明 |
|----------|------|
| Late Fusion | 各模态独立 encoder → 末端融合 → 预测头 |
| Early Fusion | 原始特征拼接 → 联合 encoder → 预测头 |

选择方式：`--backbone latefusion` 或 `--backbone earlyfusion`

### 2.3 Backbone 分离可行性

**✅ 可以分离。** CASP 的三步流程为：

```text
Step 1: pretrain（仅 backbone，标准训练）
Step 2: contrastive adaptation（TTA，更新 norm 层参数）
Step 3: pseudo-label self-training（TTA）
```

**P6 普通 baseline 比较只使用 Step 1（pretrain backbone）**，不包含 TTA 步骤。
TTA 完整流程结果单独标注为 `CASP (TTA)`。

### 2.4 技术要求

| 项目 | 内容 |
|------|------|
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.8.0 |
| 特征来源 | "Toolkit" 或 Google Drive 预提取特征 |

### 2.5 核验结论

| 维度 | 结论 |
|------|------|
| Backbone 可分离 | ✅ 可以 |
| 特征是否公开 | ⚠ 需确认 Google Drive 链接是否有效 |
| 环境兼容性 | Python ≥ 3.8, PyTorch ≥ 1.8.0（兼容） |

---

## 三、环境兼容性初步判断

### 3.1 版本对比

| 组件 | 当前 mme | CASP 要求 | MLCL（待确认） | 推荐 mme_xlstm |
|------|----------|-----------|----------------|----------------|
| Python | 3.9.21 | ≥ 3.8 | 待确认 | 3.10 |
| PyTorch | 2.3.0+cu118 | ≥ 1.8.0 | 待确认 | 2.3.0+cu118 |
| Transformers | 4.34.1 | — | 待确认 | 4.34.1+ |

### 3.2 Python 版本推荐

**推荐 Python 3.10**，理由：
1. PyTorch 2.3.0 对 Python 3.10 支持最成熟
2. 与 CASP (≥3.8) 兼容
3. 比 3.11 更稳定（3.11 在部分 Windows CUDA 库上有已知问题）
4. Conda 对 3.10 的预编译包更丰富

### 3.3 推荐环境策略

```text
mme_xlstm (Python 3.10 + PyTorch 2.3.0+cu118)  → 主模型训练
mme (Python 3.9.21)                              → 保留为只读对照
mme_mlcl (按需)                                   → MLCL baseline（如依赖冲突）
mme_casp (按需)                                   → CASP baseline（如依赖冲突）
```
