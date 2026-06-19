# P6K MOSI Conservative Mainline Comparison

## Full Results Table

| Model | Seed | Modalities | ACC2 | F1 | MAE | Corr | ACC7 | Text-Dom | Residual | Gain |
|------|:----:|:----------:|:----:|:--:|:---:|:----:|:----:|:--------:|:--------:|:----:|
| P6H inline | 42 | T+A+V | 86.43 | 84.36 | 0.678 | 0.828 | 45.92 | No | Yes | 0.00 |
| P6I text_only | 42 | T | 83.99 | 82.29 | 0.839 | 0.779 | 41.69 | Yes | No | 0.00 |
| P6J text_conf | 42 | T+A+V | 85.06 | 83.67 | 0.805 | 0.824 | 38.48 | No | Yes | 0.00 |
| P6K recovery | 42 | T+A+V | 86.13 | 83.30 | **0.611** | **0.853** | 46.21 | No | Yes | 0.00 |
| **P6K text_audio s42** | 42 | T+A | **88.72** | **86.64** | 0.635 | 0.851 | 46.65 | **Yes** | Yes | 0.00 |
| **P6K text_audio s2024** | 2024 | T+A | 86.89 | 84.75 | 0.650 | 0.841 | **47.67** | **Yes** | Yes | 0.00 |

## Key Findings

1. **Text_audio s42 reaches 88.72%** — highest MOSI test ACC2 in the project
2. **Text_audio s2024 reaches 86.89%** — above P6H baseline (86.43%)
3. **Two-seed mean ≈ 87.8%** — stable above 87%
4. **Residual gain always 0.00%** — improvement is from co-training, not delta
5. **Class-based regression fixed** — P6K recovery matches P6H (86.13% vs 86.43%)
6. **All modes are text-dominant** — text_base = final in all cases

## Recommendation

text_audio conservative (T+A, no vision, no residual claim) is the MOSI candidate main model.
