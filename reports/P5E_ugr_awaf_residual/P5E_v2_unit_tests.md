# P5E V2 Unit Tests

| # | Test | Result |
|---|------|--------|
| 1 | V2 forward (MOSI dims) | ✅ 3.67M params |
| 2 | AWAF sum=1 | ✅ max dev=5.96e-08 |
| 3 | Gate range [0,1] | ✅ [0.15, 0.18] |
| 4 | effective_delta present | ✅ |
| 5 | Delta experts (t,a,v) | ✅ All 6 experts present |
| 6 | All ablations (no_residual, no_audio, no_vision, no_awaf_mean) | ✅ |
| 7 | Gate/expert toggle | ✅ |
| 8 | MOSEI SDK dims | ✅ 1.07M params |
| 9 | Backward | ✅ |
| 10 | Loss V2 computable | ✅ delta_target=0.92, margin_sign=0.99 |
| 11 | Loss V2 backward | ✅ |
| 12 | Ablation no_residual verified | ✅ reg==reg_text_base |
