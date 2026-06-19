# P6K Textbase Regression Fix Analysis

## Root Cause Identified

P6H inline script's text_base=86.43% comes from **T+A+V multimodal co-training** (with AWAF+gate+delta), NOT from text-only training. The P6I/P6J class-based `text_only` mode removes ALL audio/vision branches, losing the co-training benefit.

## Key Differences Fixed

| Factor | P6H Original | P6J (broken) | P6K (fixed) |
|---|---|---|---|
| Mode | T+A+V full model | text_only | text_av_residual |
| Audio/Vision/AWAF | Present | Absent | Present |
| Epochs | 50 | 30/50 | 50 |
| Early stopping | None | patience=10 | patience=0 (disabled) |
| tau_init | 1.0 | 3.0 | 1.0 |
| gate_init_bias | -1.5 | +2.0 | -1.5 |
| delta_scale_init | 0.02 | 0.2 | 0.02 |
| use_modal_layernorm | N/A | True | False |
| lambda_awaf_entropy | N/A | 0.01 | 0.0 |
| lr | 5e-5 | 5e-5 | 5e-5 |
| weight_decay | 0.03 | 0.03 | 0.03 |

## Prediction

text_base should recover to ~86% if the co-training benefit is the primary factor.
