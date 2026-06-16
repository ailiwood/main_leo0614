# P4R Metric Recompute

## Key Finding: Classification Head vs Regression Sign for ACC2

| Seed | Cls-logit ACC2 | **Reg-sign ACC2** | Best Ep ACC2 |
|:--:|:--:|:--:|:--:|
| 42 | 75.00% | **76.37%** | 76.68% |
| 2024 | 75.30% | **76.83%** | 76.83% |
| 1234 | 75.46% | **76.83%** | 76.83% |

**Reg_pred sign gives ~1.4% higher ACC2_Non0 than cls_logit.**

## Analysis

1. Classification head (cls_logit) is trained with BCE loss against (label>=0) target
2. Regression head (reg_pred) is trained with L1 loss against continuous labels
3. Reg_pred sign is more aligned with the actual label sign (96.9%/94.8%/95.2% agreement)
4. **MMSA convention**: ACC2 is computed from regression prediction sign, NOT from a separate classification head

## Recommendation

**Use reg_pred sign for ACC2_Non0/F1_Non0 (aligns with MMSA standard).**
Classification head can be dropped or kept as auxiliary only.
