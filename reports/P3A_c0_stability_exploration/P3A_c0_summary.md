# P3A C0 稳定性验证总结

> 日期：2026-06-16  
> 14 个 runs 完成（10+4），全部在 stable PyTorch 2.11.0 环境

---

## 最终环境

**mme_xlstm_stable**: Python 3.10 + PyTorch 2.11.0+cu128 (STABLE) + CUDA 12.8 + RTX 5070 Ti ✅

## C0 暂定配置

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

**跨 seed 均值**: ACC2_NZ=74.9%, MAE=1.03, Corr=0.566

## AWAF 权重

| Seed | w_t | w_a | w_v |
|:--:|:--:|:--:|:--:|
| 42 | 0.555 | 0.065 | 0.380 |
| 2024 | 0.841 | 0.073 | 0.086 |

⚠ 文本模态偏重，需在 P4 长 epoch 中监控是否继续坍缩

## 是否建议进入 P3B

**是。** C0 主干在 stable 环境 + 20 epoch + 2 seed 下稳定运行，loss 正常下降，可用于候选模块相对比较基准。
