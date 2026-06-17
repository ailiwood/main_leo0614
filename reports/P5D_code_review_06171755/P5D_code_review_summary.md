# P5D Code Review Summary

> For: Web AI architecture review
> Date: 2026-06-17 17:55
> Branch: p5d-residual-stability-performance-sprint SHA: 35fafd6

## 1. Current Main Model

**DeepTextXLSTMAWAFResidual** (`models/deeptext_xlstm_awaf_residual.py`, ~450 lines)

```
Text [B,Tt,1024] → AttnPool → DeepMLP(3-layer) → reg_text_base, cls_text_base
Audio [B,Ta,768]  → sLSTM(1-layer) → AttnPool → h_a
Vision [B,Tv,768]  → sLSTM(1-layer) → AttnPool → h_v
Text_residual [B,Tt,1024] → AttnPool (NO sLSTM) → h_t_residual

AWAF(h_t_residual, h_a, h_v) → z_residual, awaf_weights [B,3]
  → delta_reg [B,1], delta_cls [B,1]

reg = reg_text_base + δ_reg * delta_reg  (δ_init=0.1, learnable)
cls = cls_text_base + δ_cls * delta_cls
```

Key: 3.39M params, ~13s/epoch on RTX 5070 Ti

## 2. Training Entry

- **One-stage**: `scripts/train_deeptext_xlstm_awaf_residual.py`
- **Two-stage**: `scripts/train_deeptext_xlstm_awaf_residual_twostage.py` (Stage1: text pretrain, Stage2: residual, Stage3: joint fine-tune)
- **Config**: `configs/models/deeptext_xlstm_awaf_residual_mosi.yaml`

## 3. Config

```yaml
hidden_dim: 256, text_mlp_hidden: 512, text_mlp_layers: 3
batch_size: 32, epochs: 60, lr: 1e-4
scheduler: reduce_on_plateau, patience: 8
delta_scale_init: 0.1
reg_loss_weight: 1.0, cls_loss_weight: 0.5
sign_consistency_weight: 0.1, delta_reg_weight: 0.05
```

## 4. Metrics

Sole implementation: `utils/metrics.py`
- ACC2_Non0 / F1_Non0 (primary, exclude zero label)
- ACC2_Has0 / F1_Has0 (secondary)
- MAE / Corr / ACC7
- reg_sign-based ACC2 used as primary (MMSA convention)

## 5. Loss Composition

`engine/losses.py` → `ResidualLossComputer`:
```
L_total = 1.0*L_reg(L1)
        + 0.5*L_cls(BCE)
        + 0.1*L_sign_consistency
        + 0.05*L_delta_reg (encourage small delta)
        + 0.0*L_awaf_entropy
        + optional: sample_reweight + focal_sign_loss
```

## 6. Residual Formula

```
reg_final = reg_text_base + δ_reg * delta_reg
where δ_reg is a learnable scalar (init=0.1)
```

P5D extension with ConditionalResidualGate:
```
reg_final = reg_text_base + g_reg(x) * δ_reg * delta_reg
where g_reg ∈ [0,1] is predicted by a gate MLP from:
  [h_text_base, z_residual, |reg_text_base|, |cls_text_base|, AWAF_entropy, |delta_reg|]
```

## 7. AWAF Position

- **Location**: Residual fusion only (not main fusion)
- **Input**: h_t_residual (text), h_a (audio), h_v (vision)
- **Output**: z_residual + sample-level weights w_t:w_a:w_v
- **Weight interpretation**: Audio currently dominant (0.44 in seed42, 0.32 in seed2024)

## 8. xLSTM/sLSTM Position

- **Audio branch**: wav2vec2 frame features → sLSTM 1-layer → h_a
- **Vision branch**: CLIP frame features → sLSTM 1-layer → h_v
- **Text branch**: NO sLSTM (DeepMLP only — P5B proved sLSTM on DeBERTa is counterproductive)
- **Text residual**: NO sLSTM by default (text_slstm_on is ablation only)

## 9. Text Branch: NO sLSTM

Confirmed by P5B evidence: 76.2% with sLSTM → 80.2% without sLSTM.
DeBERTa-large tokens are already contextualized via 24-layer disentangled attention.

## 10. Implemented but NOT Trained

| Module | File | Status |
|--------|------|--------|
| ConditionalResidualGate | models/modules/conditional_residual_gate.py | 8/8 tests pass, integrated but no training |
| Two-stage trainer | scripts/train_deeptext_xlstm_awaf_residual_twostage.py | Script ready, not run |
| Sample reweight | engine/losses.py (sample_reweight_enabled) | Implemented, not tested in training |
| Focal sign loss | engine/losses.py (sign_focal_enabled) | Implemented, not tested |

## 11. Incomplete Experiments

| Experiment | Priority | Est. Time |
|------------|----------|-----------|
| Diagnostic ablation (no_audio, no_vision, no_awaf_mean, no_residual, text_slstm_on) × 30ep | High | ~45 min |
| ConditionalResidualGate seed42 60ep | High | ~13 min |
| Two-stage training seed42 60ep | Medium | ~30 min |
| Weak_neg reweight seed42 60ep | Medium | ~13 min |

## 12. Current Performance Bottleneck

1. **2-seed mean ACC2 = 81.25%** — stable but modest gain over text-only 80.2%
2. **Residual effect is subtle**: 50.4% improved, 49.6% damaged overall
3. **Strong_neg over-corrected**: 53.3% damaged (ConditionalResidualGate should help)
4. **Val-test gap ~2-3%**: Best val ACC2 83.8%, test 81.4% — overfitting or small dataset variance
5. **Text-only ceiling 80.2%**: DeBERTa-large may be the real bottleneck for text understanding

## 13. Questions for Web AI

1. Is the residual architecture correct? Should final = text_base + δ*delta, or should we consider other fusion patterns?
2. Is ConditionalResidualGate the right direction? Will it solve the strong_neg over-correction?
3. Is two-stage training necessary or is joint training sufficient?
4. Is the current performance (81.25%) sufficient for a master's thesis, or is >83% required?
5. Should we pursue P5E strong feature upgrade (wav2vec2-large, CLIP-ViT-L)?
6. Should we consider removing vision entirely and doing text+audio only?
7. Is the 3.2% val-test gap concerning? How to reduce it?
8. Is the AWAF weight interpretation (audio-dominant residual) scientifically meaningful?
