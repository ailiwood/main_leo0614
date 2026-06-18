# P6J Textbase P6H-Equivalent Result

| Metric | P6H Original | P6J (latest code) | Δ |
|---|---|---|---|
| best_val_ACC2 | 87.50% | 86.57% | -0.93% |
| test_ACC2 | **86.43%** | **83.99%** | **-2.44%** 🔴 |
| MAE | 0.678 | 0.839 | +0.161 |
| Corr | 0.828 | 0.779 | -0.049 |
| F1_Non0 | 84.36% | 82.29% | -2.07% |
| Epochs | 50 (no early stop) | 15 (early stop E5) | |
| LoRA lr | 5e-5 | 5e-5 | same |
| Batch/Accum | 4/4 | 4/4 | same |

## Verdict

- ❌ Not fully reproduced: test 83.99% vs P6H 86.43% (gap 2.44%)
- Val close: 86.57% vs 87.50% (gap 0.93%)
- Code diff: P6H inline script vs P6I class-based
- Early stopping: P6J stops at E15 (patience=10), P6H runs all 50
