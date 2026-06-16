# P2 最低主干实现计划

> 生成时间：2026-06-16  
> 基于：P0 代码审计、《主模型实现规划.md》、《代码重构建议.md》

---

## 一、目标

实现主模型最小可运行版本：
```text
三模态投影 + 三路 sLSTM + AWAF + 多任务预测头
```

验收标准：
1. 前向不报错
2. loss 可计算、可反传
3. AWAF 输出 `w_t + w_a + w_v = 1`
4. 权重可落盘
5. 真实数据 1 epoch smoke test

---

## 二、需要新建的文件

### 2.1 核心模块

| 文件 | 内容 | 优先级 |
|------|------|:--:|
| `models/encoders/slstm.py` | `SLSTMCell` + `SLSTMEncoder` | P0 |
| `models/fusion/awaf.py` | `AdaptiveWeightedAttentionFusion` | P0 |
| `models/heads.py` | 回归头 + 分类头 + 单模态辅助头 | P0 |
| `models/ours_xlstm_fusion.py` | 主模型组装 | P0 |
| `models/base.py` | BaseModel 基类 | P1 |

### 2.2 基础设施

| 文件 | 内容 | 优先级 |
|------|------|:--:|
| `utils/metrics.py` | 统一指标 (ACC2_Non0/F1_Non0/MAE/Corr/ACC7/Has0) | P0 |
| `utils/seed.py` | 随机种子控制 | P0 |
| `configs/default.yaml` | 默认配置 | P0 |
| `configs/models/ours_backbone.yaml` | 主模型配置 | P0 |

### 2.3 数据和训练

| 文件 | 内容 | 优先级 |
|------|------|:--:|
| `data/dataset.py` | 统一数据集类 | P0 |
| `engine/trainer.py` | 最小训练器 | P0 |
| `engine/registry.py` | 模型注册 | P1 |

---

## 三、sLSTM 自实现要求

基于 P0 审计结论（`lstm_v.py` 为 mLSTM + AGPL-3.0，不可复用），sLSTM 必须**完全自实现**：

### 3.1 SLSTMCell

```python
class SLSTMCell(nn.Module):
    """
    sLSTM 单元（纯 PyTorch 自实现）

    输入: x_t [B, D]
    输出: h_t [B, H]

    包含：指数门控、normalizer state、stabilizer state、mask 更新、NaN 防护
    """
```

参考来源：Beck et al. (2024) xLSTM paper (NeurIPS)，公式推导自论文，不复制 NX-AI AGPL 代码。

### 3.2 SLSTMEncoder

```python
class SLSTMEncoder(nn.Module):
    """
    三模态时序编码器

    输入: x [B, T, D], mask [B, T]
    输出: H [B, T, H]

    支持：masked mean / last_valid pooling
    """
```

---

## 四、AWAF 自实现要求

### 4.1 模块结构

```python
class AdaptiveWeightedAttentionFusion(nn.Module):
    """
    样本级自适应加权注意力融合

    第一段：跨模态上下文增强 (Context Enhancement)
    第二段：二阶 Hadamard 交互项生成样本级权重 (Interaction Scoring)

    输入: h_t, h_a, h_v ∈ R^d
    输出:
      - Z ∈ R^d (融合表示)
      - w = [w_t, w_a, w_v] (样本级三模态权重，sum(w)=1)
    """
```

### 4.2 消融开关

| 模式 | 含义 |
|------|------|
| `awaf` | 完整 AWAF（默认） |
| `awaf_no_interaction` | 去掉二阶交互项 g_ta/g_tv/g_av |
| `awaf_no_context` | 去掉跨模态上下文增强 |
| `mean` | 等权平均 |
| `gated` | 普通门控融合 |
| `concat` | 拼接 + MLP |
| `fixed` | 全局固定可学习权重 |
| `modality_dropout_off` | 关闭模态 dropout |

---

## 五、执行步骤

1. **先写 `utils/metrics.py`**（所有实验的共同依赖）
2. **写 `utils/seed.py`**（确定性训练的基础）
3. **写 `models/encoders/slstm.py`**：`SLSTMCell` → `SLSTMEncoder`
4. **写 `models/fusion/awaf.py`**：含全部消融开关
5. **写 `models/heads.py`**：回归头/分类头/辅助头
6. **写 `models/ours_xlstm_fusion.py`**：组装主模型
7. **写 `configs/default.yaml`**：集中配置
8. **随机张量 forward test**：验证形状、权重、sum(w)=1
9. **等 P1 特征就绪后**：真实数据 1 epoch smoke test

---

## 六、smoke test 标准

```text
1. forward pass 不报错
2. loss 可计算且有限
3. backward pass 不报错
4. AWAF 输出 w_t + w_a + w_v ≈ 1.0 (tolerance 1e-5)
5. AWAF 权重可保存为 CSV
6. 不同消融开关可切换
7. 无演示数据或随机特征冒充真实实验
```
