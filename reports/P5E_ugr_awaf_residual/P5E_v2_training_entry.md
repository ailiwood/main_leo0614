# P5E V2 Training Entry

## Script

`scripts/train_deeptext_xlstm_awaf_residual_v2.py`

## Supported Features

| Feature | Flag | Default |
|---------|------|---------|
| UGR Gate | `use_uncertainty_gate` | true |
| Delta Experts | `use_delta_experts` | true |
| Bounded Delta | `use_bounded_delta` | true |
| Delta Target Loss | `delta_target_loss_weight` | 0.2 |
| Margin Sign Loss | `margin_sign_loss_weight` | 0.05 |
| Sample Reweight | `sample_reweight_enabled` | false |

## Saved Outputs

- `predictions_test.csv` ✅
- `awaf_weights_test.csv` ✅
- `text_base_delta_gate_test.csv` ✅ (includes gate_reg, effective_delta_reg)
- `best_model.pth` ✅ (only one checkpoint)
- No `last_model.pth` ❌ (P5D policy)

## V2Trainer

Custom trainer using ResidualLossV2 with per-sample reduction='none'.
