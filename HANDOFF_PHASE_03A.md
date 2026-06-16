# HANDOFF_PHASE_03A.md

> 阶段：P3A — C0 主干稳定性验证与超参数探索  
> 日期：2026-06-16  
> 状态：✅ 完成

---

## 1. 本阶段使用的最终环境

| 项目 | 值 |
|------|-----|
| 环境名 | **mme_xlstm_stable** ✅ |
| Python | 3.10.20 |
| PyTorch | **2.11.0+cu128 (STABLE)** |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| Capability | sm_120 ✅ native |

**建议后续继续使用 stable 环境。** 旧 mme_xlstm (nightly) 保留为备份。

---

## 2. 本阶段运行了哪些实验

### 2.1 Seed=42 (10 组合 × 20 epoch)

| # | Run ID | ACC2_NZ | MAE | Corr | Note |
|---|--------|:--:|:--:|:--:|------|
| 1 | h128_l2_lr5e-5_d0.3 | 74.24% | 1.130 | 0.520 | Best @ s=42 |
| 2 | h256_l1_lr1e-4_d0.3 | 73.48% | 1.058 | 0.537 | ★ Most stable |
| 3 | h256_l1_lr5e-5_d0.3_bs16 | 73.32% | 1.089 | 0.526 | |
| 4 | h128_l1_lr5e-5_d0.3 | 72.71% | 1.117 | 0.509 | Efficient |
| 5 | h256_l1_lr5e-5_d0.1 | 72.26% | 1.088 | 0.513 | |
| 6 | h256_l1_lr5e-5_d0.3_aux0.1 | 71.95% | 1.143 | 0.489 | |
| 7 | h256_l1_lr5e-5_d0.3 | 71.19% | 1.097 | 0.505 | |
| 8 | h256_l1_lr5e-5_d0.3_md0 | 70.43% | 1.187 | 0.515 | |
| 9 | h256_l2_lr5e-5_d0.3 | 70.12% | 1.194 | 0.526 | |
| 10 | h128_l1_lr5e-5_d0.1 | 67.99% | 1.267 | 0.469 | |

### 2.2 Seed=2024 (Top 4 验证 × 20 epoch)

| # | Run ID | ACC2_NZ | MAE | Corr |
|---|--------|:--:|:--:|:--:|
| 1 | **h256_l1_lr1e-4_d0.3** | **76.37%** | 1.003 | 0.594 |
| 2 | h256_l1_lr5e-5_d0.3_bs16 | 73.78% | 1.070 | 0.544 |
| 3 | h128_l2_lr5e-5_d0.3 | 70.73% | 1.240 | 0.494 |
| 4 | h128_l1_lr5e-5_d0.3 | 70.12% | 1.184 | 0.494 |

---

## 3. 最佳 C0 配置

```yaml
hidden_dim: 256
slstm_num_layers: 1
slstm_pooling: masked_mean
awaf_fusion_mode: awaf
awaf_modality_dropout: true
dropout: 0.3
lr: 1e-4
batch_size: 32
aux_loss_weight: 0.0
```

**选择理由**: 跨 seed 最稳定 (均值 74.9%)，两个 seed 均为 top 性能，单层简洁。

---

## 4. AWAF 权重分析

- 文本模态偏重 (w_t=0.55-0.84)
- 权重随 seed 变化（同一 config 不同 init 权重分布不同）
- **未塌缩到全 0/1**（w_a, w_v 虽有波动但仍参与融合）
- P4 长 epoch 需监控是否趋向退化

---

## 5. 仍然存在的风险

1. **T=1 特征限制** — 所有结论基于 clip-level 单向量
2. **MOSI 单数据集** — MOSEI 未验证
3. **AWAF 权重偏重文本** — 可能随 epoch 增加坍缩
4. **20 epoch 可能不足** — P4 需 50+ epoch

---

## 6. 下一步建议

进入 **P3B**：候选模块裁决实验 (C1-C3)。

优先：
1. **C1: C0 + DEConv** — 视觉动态增强，实现最直接
2. **C3: C0 + CME** — 跨模态交互，但需先审计 CME 是否依赖 AGPL
3. **C2: C0 + Data2Vec-Audio** — 音频特征替换，需评估特征提取成本
