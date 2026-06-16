# HYPERPARAMS.md — C0 最佳超参数记录

> 产出：P3A  
> 日期：2026-06-16  
> 数据集：MOSI TMDC-v1 (T=1)

---

## C0 暂定最佳配置

| 超参 | 值 | 选择理由 |
|------|-----|------|
| hidden_dim | **256** | 256 > 128 一致 (约 2%) |
| slstm_num_layers | **1** | 2层在 256 维反而差 |
| dropout | **0.3** | 0.3 > 0.1 一致 |
| lr | **1e-4** | 优于 5e-5 约 1-2% |
| batch_size | **32** | 16 略好但不稳定，32 更高效 |
| aux_loss_weight | **0.0** | 无明显帮助 |
| awaf_modality_dropout | **true** | ON > OFF 约 1% |
| awaf_fusion_mode | **awaf** | 完整 AWAF |
| slstm_pooling | **masked_mean** | 默认 |
| weight_decay | **0.01** | 标准 |

## 探索历史

- 共运行 14 个 20-epoch runs
- 搜索了 hidden_dim (128/256), layers (1/2), lr (5e-5/1e-4), dropout (0.1/0.3), batch_size (16/32), aux (0/0.1), modality_dropout (T/F)
- 2 seeds (42, 2024)

## 跨 seed 表现

| Seed | ACC2_NZ | MAE | Corr |
|:--:|:--:|:--:|:--:|
| 42 | 73.48% | 1.058 | 0.537 |
| 2024 | 76.37% | 1.003 | 0.594 |
| 均值 | 74.9% | 1.031 | 0.566 |

## 待 P4 验证

- MOSEI 数据集
- 更多 seed (≥3)
- ≥50 epoch
- 真正的序列特征 (T > 1)
