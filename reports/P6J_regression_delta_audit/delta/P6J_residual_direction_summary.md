# P6J Residual Direction Audit — P6I text_confidence_residual

## Key Metrics

| Metric | Value | Judgment |
|---|---|---|
| delta_sign_correct_rate | **48.69%** | 🔴 ≈random (50%) |
| error_reduced_rate | 43.59% | 🔴 <50% |
| text_wrong_final_right | 8 | 🟡 few fixes |
| text_right_final_wrong | 18 | 🔴 more breaks |
| **net_correct_gain** | **-10** | 🔴 **residual HARMFUL** |
| text_base ACC2 | 85.28% | |
| final ACC2 | 83.82% | 🔴 **worse** |
| delta_abs_mean | 0.234 | large |
| pred_delta_abs_mean | 0.234 | |
| target_delta_abs_mean | 0.742 | |

## Per-Group Analysis

| Group | N | tb_acc | fn_acc | TWFR | TRFW | net | sign% | err↓% |
|---|---|---|---|---|---|---|---|---|
| strong_neg | 202 | 90.6% | 89.1% | 1 | 4 | **-3** | 19.8% | 16.8% |
| weak_neg | 165 | 78.2% | 72.1% | 1 | 11 | **-10** | 45.5% | 37.0% |
| near_zero | 64 | 65.6% | 64.1% | 2 | 3 | **-1** | 62.5% | 53.1% |
| weak_pos | 135 | 84.4% | 87.4% | 4 | 0 | **+4** | 80.0% | 78.5% |
| strong_pos | 120 | 97.5% | 97.5% | 0 | 0 | 0 | 59.2% | 53.3% |

- **weak_neg 最受害**: net -10 (11 个对的被改错)
- **weak_pos 唯一受益**: net +4 (4 个错的被改对)
- **near_zero**: sign_correct 62.5% (最高) 但 net 仍 -1

## Gate vs Error Correction

| Gate Range | N | sign_correct | error_reduced |
|---|---|---|---|
| 0.00-0.50 | 91 | **74.7%** | **72.5%** |
| 0.50-0.70 | 102 | 71.6% | 66.7% |
| 0.70-0.85 | 194 | 33.0% | 28.4% |
| 0.85-1.00 | 299 | 43.1% | 36.8% |

> **高 gate 时 delta 方向最差！** Gate 打开→delta 大→方向错误→破坏 text_base。

## Key Judgments

| # | Judgment | Value |
|---|---|---|
| 1 | net_correct_gain | **-10 < 0** → residual HARMFUL |
| 2 | delta_sign_correct_rate | **48.69% ≈ 50%** → direction RANDOM |
| 3 | error_reduced_rate high but ACC worse | ✅ MAE 改善但分类边界破坏 |
| 4 | weak_pos 受益, weak_neg 受害 | asymmetric residual effect |

## Root Cause

Delta MLP 无法从 A/V features 学习正确的 directional correction:
- A/V signal 太弱 (audio=56.9%, vision=65.3%)
- 无法为 text error 提供一致的修正方向
- Delta 学习到的是 noisy A/V patterns, not meaningful error corrections
