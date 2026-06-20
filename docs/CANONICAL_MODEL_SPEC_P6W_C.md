# P6W-C Canonical Model Specification

## Model Name
`canonical_text_audio_awaf_slstm`

## Mode
```yaml
mode: canonical_text_audio_awaf_slstm
```

## Architecture (immutable)
1. Text: RoBERTa-large → LoRA → MLP → h_t [B, H]
2. Audio: COVAREP 74d → Linear(LN,GELU) → sLSTM → MaskedAttentionPool → h_a [B, H]
3. Fusion: AWAF(h_t, h_a) → z [B, H] with w_t + w_a = 1
4. Head: Linear(H→H/2)→ReLU→Linear(H/2→1) applied to z
5. Output: reg = head(z)

## Forbidden
- NO text bypass (reg=rtb)
- NO gate=off→rtb path
- NO delta=0→rtb path
- NO vision zeros placeholder
- NO allow_text_bypass

## Config Fields
```yaml
fusion_type: awaf
temporal_encoder: slstm
awaf_context: true
awaf_interaction: true
use_uncertainty_gate: false
use_delta_experts: false
allow_text_bypass: false
save_awaf_weights: true
```

## Verification
- AWAF must be created (needs_awaf=True)
- text AND audio must enter AWAF
- audio must go through projection→sLSTM→pooling
- final prediction from z, not rtb
- gate/delta default off
