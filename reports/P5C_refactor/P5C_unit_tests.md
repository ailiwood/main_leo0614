# P5C Unit Tests Report

> Date: 2026-06-17
> Model: DeepText-xLSTM-AWAF Residual
> Device: CUDA (RTX 5070 Ti)

## Test Results Summary

| # | Test | Result | Details |
|---|------|--------|---------|
| 1 | MOSI dims forward/backward | ✅ PASS | 1024/768/768, 3.39M params, No NaN |
| 2 | MOSEI SDK dims forward/backward | ✅ PASS | 300/74/35, 0.98M params, No NaN |
| 3 | AWAF sum(w)=1 | ✅ PASS | max deviation = 0.00e+00 across 5 batches |
| 4 | Padding invariance | ✅ PASS | Valid output with padding, no NaN |
| 5 | no_residual ablation | ✅ PASS | reg == reg_text_base exactly (diff=0.0) |
| 6 | no_audio / no_vision | ✅ PASS | Both produce valid outputs |
| 7 | text_slstm_on ablation | ✅ PASS | Forward+backward OK, 3.92M params |
| 8 | delta_scale learnable | ✅ PASS | δ_reg: 0.1000→0.0341, δ_cls: 0.1000→0.5193 |
| 9 | Loss computable | ✅ PASS | All 7 loss components computable |

**Total: 9/9 passed**

## Key Observations

### Model Size
- Full model (MOSI): 3,392,270 params
- With text_slstm_on: 3,917,582 params (+525K)
- MOSEI SDK dims: 977,422 params

### Delta Scale Dynamics
After 1 SGD step (lr=0.5):
- δ_reg: 0.1000 → 0.0341 (decreased — model prefers smaller regression correction)
- δ_cls: 0.1000 → 0.5193 (increased — model wants larger classification correction)

This is interesting: the model naturally learns different scales for regression vs classification corrections.

### AWAF Weights
In eval mode with random inputs: w_t=0.2056, w_a=0.1780, w_v=0.6164
Vision gets the highest weight in the residual branch.
