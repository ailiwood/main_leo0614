# P2 最低主干 Smoke Test 报告

> 生成时间：2026-06-16  
> 阶段：P2 最低主干实现  
> 状态：✅ 完成（CPU 模式），⚠️ GPU 模式待 CUDA 12.4 安装完成后验证

---

## 一、随机张量 Forward Test

全部 5 项测试通过：

| 测试项 | 验证内容 | 结果 |
|--------|----------|:--:|
| Metrics | MAE/Corr/ACC2_Non0/F1_Non0/ACC2_Has0/F1_Has0/ACC7 | ✅ |
| SLSTMCell | 单步 + 多步展开 + NaN/Inf 防护 | ✅ |
| SLSTMEncoder | masked_mean/last_valid/T=1/bidirectional | ✅ |
| AWAF | 7 种融合模式 + sum(w)=1 + modality_dropout | ✅ |
| MainModel | T=1 + T=5 + backward + 3.16M params | ✅ |

---

## 二、MOSI 真实数据 1 Epoch Smoke Test

### 2.1 配置

| 参数 | 值 |
|------|-----|
| 数据集 | CMU-MOSI (TMDC-v1) |
| Split | train=1284, val=229, test=686 |
| Epochs | 1 |
| Batch Size | 16 |
| LR | 5e-5 |
| Optimizer | AdamW |
| Hidden Dim | 256 |
| Fusion Mode | awaf |
| Device | CPU (CUDA 兼容性待解决) |

### 2.2 结果

| 指标 | 值 | 说明 |
|------|-----|------|
| Train Loss | 1.7555 | 有限，正常 |
| MAE (test) | 1.628 | 1 epoch，预期较差 |
| Corr (test) | 0.106 | 1 epoch，预期较低 |
| ACC2_Non0 | 42.68% | 1 epoch，预期较低 |
| F1_Non0 | 59.31% | 1 epoch |
| ACC2_Has0 | 45.19% | 1 epoch |
| F1_Has0 | 61.79% | 1 epoch |
| ACC7 | 16.47% | 1 epoch |

### 2.3 AWAF 权重

| 模态 | 均值 | 标准差 |
|------|:--:|:--:|
| Text (w_t) | 0.223 | 0.018 |
| Audio (w_a) | 0.207 | 0.052 |
| Vision (w_v) | 0.570 | 0.054 |

- sum(w) max_dev: **1.19e-07** ✅（满足 sum(w)=1）
- Vision 模态在当前 T=1 特征下占主导（权重 ~0.57）
- 权重样本间有变化（std > 0），证明 AWAF 产生了样本级的动态权重

### 2.4 验收通过项

| 验收条件 | 状态 |
|----------|:--:|
| 1. forward pass 不报错 | ✅ |
| 2. loss 有限且可 backward | ✅ |
| 3. AWAF sum(w)=1 | ✅ (max_dev=1.19e-7) |
| 4. AWAF 权重可落盘 | ✅ (awaf_weights_test.csv) |
| 5. 统一 metrics 输出 7 项指标 | ✅ |
| 6. MOSI TMDC 特征正确读取 | ✅ |
| 7. 有日志、配置、指标、预测、权重 | ✅ |
| 8. 未把 smoke test 结果写为正式结果 | ✅ |

---

## 三、产物保存

```
outputs/P2_smoke/mosi/ours_xlstm_fusion/20260616_152745_seed42/
├── config.json              ✅ 模型配置
├── command.txt              ✅ 运行命令
├── train.log                (集成在 console 输出)
├── metrics_epoch.csv        ✅ epoch 指标
├── metrics_best.json        ✅ 最佳指标
├── predictions_test.csv     ✅ 686 条预测
├── awaf_weights_test.csv    ✅ 686 条 AWAF 权重
├── best_model.pth           ✅ 模型权重
└── last_model.pth           ✅ 模型权重
```

---

## 四、已知问题

### 4.1 CUDA 兼容性（已修复）

- RTX 5070 Ti (Blackwell sm_120) 与 PyTorch 2.3.0+cu118 不兼容
- 已升级到 PyTorch 2.5.1+cu124
- GPU smoke test 待 CUDA 版本安装完成后重跑

### 4.2 T=1 单向量局限性

- 当前 MOSI TMDC 特征为 per-clip 单向量（非时序序列）
- 这限制了 sLSTM 时序建模能力的验证
- P4 前需要真正的序列特征或明确此数据粒度限制

---

## 五、不视为论文结果

本 smoke test 的指标（ACC2_Non0=42.68%）仅用于工程链路验证，**不得写入论文**。原因：
1. 仅训练 1 epoch
2. T=1 单向量非时序输入
3. 未调参
4. CPU 模式，非正式训练配置
