# P4A MOSI C0 正式训练汇总

## 最终 C0 配置

```yaml
hidden_dim: 256, slstm_layers: 1, dropout: 0.3
lr: 1e-4, batch_size: 32, epochs: 60
modality_dropout_prob: 0.2  (stronger, from ablation)
awaf_entropy_reg_weight: 0.0  (not adopted)
```

## AWAF 正则化裁决

| Config | ACC2_NZ | vs base |
|--------|:--:|:--:|
| C0_base (md=0.1) | 73.63% | — |
| ent_0.01 | 74.24% | +0.61% |
| ent_0.05 | 73.40% | -0.23% |
| **stronger_md (md=0.2)** | **75.61%** | **+1.98%** ✅ |
| ent_0.01+md_0.2 | 74.09% | +0.46% |

**Decision**: Adopt md=0.2, no entropy regularization.

## MOSI 3-Seed 60-Epoch Results

| Seed | ACC2_NZ | MAE | Corr | F1_NZ | ACC7 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 42 | 75.00% | 1.009 | 0.584 | 68.58 | 32.65 |
| 2024 | 75.30% | 0.992 | 0.601 | 70.55 | 32.51 |
| 1234 | 75.46% | 0.999 | 0.592 | 69.33 | 32.07 |
| **Mean** | **75.25%** | **1.000** | **0.592** | **69.49** | **32.41** |
| **Std** | **0.23%** | **0.008** | **0.009** | | |

### AWAF 权重

| Seed | w_t | w_a | w_v | entropy |
|:--:|:--:|:--:|:--:|:--:|
| 42 | .679 | .061 | .260 | 0.628 |
| 2024 | .554 | .072 | .375 | 0.607 |
| 1234 | .473 | .130 | .396 | 0.710 |

w_t 未坍缩到 0.9+，w_v 保持在 0.26-0.40 ✅

## 关键比较

| Comparison | ACC2_NZ Δ |
|------------|:--:|
| vs P3C C0_40ep base (md=0.1) | **+1.62%** |
| vs P3C C1_40ep (DEConv) | +2.23% |
| vs P3C C3 (CME) | +2.23% |

## 科学边界

⚠ T=1 clip-level 特征，不是论文正式结果。MOSEI 未完成。
