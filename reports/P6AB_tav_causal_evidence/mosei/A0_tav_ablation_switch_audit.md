# P6AB TAV Ablation Switch Integrity Audit

**Date**: 2026-06-21
**Device**: cpu
**Seed**: 42

## Switch Integrity Checks

| Check | Status | Detail |
|-------|--------|--------|
| F1_no_vision_branch | PASS pass | Vision branch: absent (OK) |
| F2_no_audio_branch | PASS pass | Audio branch: absent (OK) |
| F0_vs_F1_outputs_differ | PASS pass | |F0_mean - F1_mean| = 0.386774 |
| F0_vs_F2_outputs_differ | PASS pass | |F0_mean - F2_mean| = 0.356253 |
| F3_fixed_fusion | PASS pass | Fusion mode: fixed |
| F4_mean_fusion | PASS pass | Fusion mode: mean |
| F5_no_interaction | PASS pass | Use interaction: False |
| E1_lstm_encoder | PASS pass | Temporal encoder: lstm |
| all_outputs_not_identical | PASS pass | Unique reg_mean values: 7/7 |

**Overall verdict**: PASS ¡ª all variants valid for training

## Model Signatures

| Variant | Params (Total/Trainable) | Audio Branch | Vision Branch | Fusion | Temporal |
|---------|--------------------------|--------------|---------------|--------|----------|
| F0 | 360,198,925 / 4,839,181 | YES | YES | awaf | slstm |
| F1 | 359,456,780 / 4,097,036 | YES | NO | awaf | slstm |
| F2 | 359,620,364 / 4,260,620 | NO | YES | awaf | slstm |
| F3 | 358,947,597 / 3,587,853 | YES | YES | fixed | slstm |
| F4 | 358,947,591 / 3,587,847 | YES | YES | mean | slstm |
| F5 | 359,805,709 / 4,445,965 | YES | YES | awaf_no_interaction | slstm |
| E1 | 360,332,557 / 4,972,813 | YES | YES | awaf | lstm |

## Fixed Batch Forward Outputs

| Variant | reg_mean | reg_std | w_t | w_a | w_v | w_unique |
|---------|----------|---------|-----|-----|-----|----------|
| F0 | 0.2185 | 0.0065 | 0.3443 | 0.3535 | 0.3022 | 4 |
| F1 | -0.1683 | 0.0157 | 0.3712 | 0.2923 | 0.3364 | 4 |
| F2 | -0.1378 | 0.0275 | 0.3670 | 0.2761 | 0.3569 | 4 |
| F3 | 0.0217 | 0.0197 | 0.3333 | 0.3333 | 0.3333 | 1 |
| F4 | -0.1605 | 0.0151 | 0.3333 | 0.3333 | 0.3333 | 1 |
| F5 | 0.0270 | 0.0090 | 0.2841 | 0.3674 | 0.3484 | 4 |
| E1 | 0.1084 | 0.0160 | 0.3571 | 0.3328 | 0.3100 | 4 |
