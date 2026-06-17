# P5D Key Code Excerpts for Web AI Review

> For architecture risk assessment without opening all files.

## 1. DeepTextXLSTMAWAFResidual.forward (core logic)

```python
# models/deeptext_xlstm_awaf_residual.py

def forward(self, text, audio, vision, text_mask=None, audio_mask=None, vision_mask=None):
    B = text.size(0)

    # --- 1. Text Branch (main discriminant, NO sLSTM) ---
    text_out = self._process_text_branch(text, text_mask)
    reg_text_base = text_out['reg_text_base']  # [B, 1]
    cls_text_base = text_out['cls_text_base']  # [B, 1]
    h_text_base = text_out['h_text_base']      # [B, H]

    # --- 2. Audio/3. Vision (residual enhancement, WITH sLSTM) ---
    h_a = self._process_audio_branch(audio, audio_mask)    # [B, H]
    h_v = self._process_vision_branch(vision, vision_mask)  # [B, H]

    # --- 4. Text Residual (lightweight, NO sLSTM) ---
    h_t_residual = self._process_text_residual_branch(text, text_mask)  # [B, H]

    # --- 5. AWAF Residual Fusion ---
    awaf_out = self.awaf(h_t_residual, h_a, h_v)  # {Z, weights}
    z_residual = awaf_out['Z']                     # [B, H]
    awaf_weights = awaf_out['weights']             # [B, 3]
    delta_reg = self.delta_reg_head(z_residual)    # [B, 1]
    delta_cls = self.delta_cls_head(z_residual)    # [B, 1]

    # --- 6. Conditional Residual Gate (P5D) ---
    if self.use_residual_gate:
        gate_out = self.residual_gate(
            h_text_base, z_residual, reg_text_base, cls_text_base,
            awaf_weights, delta_reg)
        reg = reg_text_base + gate_out['gate_reg'] * self.delta_scale_reg * delta_reg
        cls = cls_text_base + gate_out['gate_cls'] * self.delta_scale_cls * delta_cls
    else:
        reg = reg_text_base + self.delta_scale_reg * delta_reg
        cls = cls_text_base + self.delta_scale_cls * delta_cls
```

## 2. Residual Final Prediction Formula

```
Without gate:  reg = reg_text_base + δ_reg * delta_reg
With gate:     reg = reg_text_base + g_reg(x) * δ_reg * delta_reg

δ_reg, δ_cls are learnable nn.Parameters (init=0.1)
g_reg, g_cls ∈ [0, 1] from ConditionalResidualGate MLP
```

## 3. AWAF Call Location

```python
# Called inside forward, AFTER text_base is computed
awaf_out = self.awaf(h_t_residual, h_a, h_v)  # AdaptiveWeightedAttentionFusion
z_residual = awaf_out['Z']      # [B, H] — fused residual representation
awaf_weights = awaf_out['weights']  # [B, 3] — w_t, w_a, w_v per sample
```

## 4. delta_scale Definition

```python
# models/deeptext_xlstm_awaf_residual.py __init__:
self.delta_scale_reg = nn.Parameter(torch.tensor(0.1))  # init=0.1
self.delta_scale_cls = nn.Parameter(torch.tensor(0.1))
# These are learnable — optimizer will adjust them
```

## 5. ConditionalResidualGate

```python
# models/modules/conditional_residual_gate.py
class ConditionalResidualGate(nn.Module):
    def __init__(self, hidden_dim, gate_hidden_dim=128, init_bias=-1.0):
        # Input features:
        #   h_text_base [B,H] + z_residual [B,H] → 2H
        #   + |reg_text_base| [B,1] + |cls_text_base| [B,1]  (text confidence)
        #   + AWAF_entropy [B,1]  (awaf uncertainty)
        #   + |delta_reg| [B,1]   (delta magnitude)
        # → input_dim = 2H + 2 + 1 + 1 = 2H+4
        self.gate_net = nn.Sequential(
            Linear(input_dim → 128) → LN → GELU → Dropout
            → Linear(128 → 64) → LN → GELU → Dropout
            → Linear(64 → 2)  # [gate_reg_raw, gate_cls_raw]
        )
        # Last layer bias = -1.0 → sigmoid(-1) ≈ 0.27 initial gate

    def forward(self, h_text_base, z_residual, reg_text_base, cls_text_base,
                awaf_weights=None, delta_reg=None):
        features = [h_text_base, z_residual]
        if self.use_text_confidence:
            features.extend([reg_text_base.abs(), cls_text_base.abs()])
        if self.use_awaf_entropy and awaf_weights is not None:
            entropy = -(awaf_weights.clamp(eps) * torch.log(awaf_weights.clamp(eps))).sum(-1, keepdim=True)
            features.append(entropy)
        if self.use_delta_magnitude and delta_reg is not None:
            features.append(delta_reg.abs())
        feat = torch.cat(features, dim=-1)
        gate_raw = self.gate_net(feat)
        return {'gate_reg': sigmoid(gate_raw[:,0:1]),
                'gate_cls': sigmoid(gate_raw[:,1:2])}
```

## 6. ResidualLossComputer (key parts)

```python
# engine/losses.py
class ResidualLossComputer:
    def compute(self, output, labels):
        # 1. Regression Loss (L1)
        reg_loss = self.reg_loss_fn(output['reg'], labels)

        # 2. Classification Loss (BCE)
        cls_loss = self.cls_loss_fn(output['cls'], polarity)

        # 3. Sign Consistency (reg_pred sign vs label sign)
        sign_logit = 2.0 * output['reg'].view(-1)  # scale for sigmoid
        sign_loss = BCE(sign_logit, polarity_target)  # optionally focal

        # 4. Delta Regularization (encourage small delta)
        delta_reg_loss = output['delta_reg'].abs().mean()

        # 5. AWAF Entropy Regularization (prevent weight collapse)
        entropy = -sum(w * log(w))

        total = w_reg*reg_loss + w_cls*cls_loss + w_sign*sign_loss
              + w_aux*aux_loss + w_delta*delta_reg_loss + entropy_reg
```

## 7. Sample Reweight Logic

```python
# engine/losses.py _compute_sample_weights:
def _compute_sample_weights(self, labels):
    weights = torch.ones_like(labels) * strong_sample_weight (1.0)
    weights[labels in (-1.0, 0)] *= weak_neg_weight (1.5)
    weights[labels in (0, 1.0)]  *= weak_pos_weight (1.2)
    weights[abs(labels) <= 0.5]  *= near_zero_weight (1.2)
    return weights
    # Applied to reg_loss: reg_loss = (per_element_l1 * weights).mean()
```

## 8. Checkpoint Save Logic

```python
# engine/strict_trainer.py save_run:
def save_run(self, out_dir, config, cmd, epoch, test_result, save_last_pth=False):
    # Always save: config.json, command.txt, val_metrics_epoch.csv,
    #   test_metrics_final.json, predictions_test.csv
    #   awaf_weights_test.csv, text_base_delta_test.csv (P5D)
    #   val_curves.png
    torch.save(best_state, 'best_model.pth')  # always
    if save_last_pth:
        torch.save(model.state_dict(), 'last_model.pth')  # P5D: default OFF
```

## 9. Strict Protocol val/test Separation

```python
# engine/strict_trainer.py:
# train_epoch: trains on train_loader ONLY
# check_val: evaluates on val_loader, updates best checkpoint by val MAE
# final_test: loads best checkpoint (selected by val), evaluates test ONCE
#   → test is NEVER used for epoch selection
```
