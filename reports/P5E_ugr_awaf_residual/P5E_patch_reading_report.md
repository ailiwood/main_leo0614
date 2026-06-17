# P5E Patch Reading Report

## Files Provided by Web AI

| File | Status | Key Content |
|------|--------|-------------|
| `docs/模型架构优化建议06171755.md` | EXISTS ✅ | Architecture optimization guide: UGR gate, delta experts, delta target loss, bounded delta |
| `docs/核心文件替换说明.md` | MISSING ❌ | Not provided; CC will implement directly |
| `models/modules/uncertainty_residual_gate.py` | EXISTS ✅ | UncertaintyGuidedResidualGate with gate_prior, blend mode, learnable margin/temperature |
| `models/deeptext_xlstm_awaf_residual_v2.py` | EXISTS ✅ | DeepTextXLSTMAWAFResidualV2 with delta experts, bounded delta, UGR gate |
| `engine/residual_losses_v2.py` | EXISTS ✅ | ResidualLossV2 with per-sample reduction='none', delta target loss, margin sign loss |
| `configs/models/deeptext_xlstm_awaf_residual_v2_mosi.yaml` | EXISTS ✅ | V2 MOSI config with delta_target=0.2, bounded_delta=true, max_delta=1.5 |

## Key Design Decisions from Web AI

1. **UGR Gate**: `gate = learned_gate * gate_prior`, where gate_prior = sigmoid((margin - |reg|)/temperature)
2. **Delta Experts**: `delta_awaf = w_t*delta_t + w_a*delta_a + w_v*delta_v` — AWAF directly weights modality experts
3. **Bounded Delta**: `max_delta * tanh(delta_raw)` — prevents over-correction
4. **Delta Target Loss**: `SmoothL1(effective_delta, label - detach(reg_text_base))` — explicit residual supervision
5. **Sample Reweight**: Uses `reduction='none'` with proper per-sample weighted mean
6. **Margin Sign Loss**: `ReLU(margin - sign * reg)` for nonzero samples

## Code Fixes Needed

1. Attention pooling return expression — check and fix
2. V2 losses already has proper per-sample reduction='none'
3. Two-stage trainer needs Stage1 best state restoration
4. New V2 training entry script needed
