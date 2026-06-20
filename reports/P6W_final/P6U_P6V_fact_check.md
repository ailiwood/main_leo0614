# P6W: P6U + P6V Fact Check for Fair Ablation

**Date**: 2026-06-20 | **Branch**: `p6w-fair-ablation-final-release`

## Confirmed Facts

| # | Fact | Status |
|---|------|--------|
| 1 | P6U freeze results NOT modified | ✅ |
| 2 | MOSI main P6K text_audio s42 = 88.72% | ✅ |
| 3 | MOSEI main P6T text_audio s42 = 87.86% | ✅ |
| 4 | MOSEI text_only is diagnostic only | ✅ |
| 5 | MOSEI TAV excluded (vision all-zero) | ✅ |
| 6 | MMIM-lite excluded (MISA duplicate) | ✅ |
| 7 | P6V smoke outputs excluded from formal tables | ✅ |
| 8 | P6V MOSI collapse (42.23%) excluded | ✅ |
| 9 | P6V MOSEI 4-epoch results → preliminary_unfair_budget | ✅ |
| 10 | P6W controls use 20ep (MOSI) / 12ep (MOSEI) = same as P6U freeze | ✅ |

## P6V Result Reclassification

| P6V Result | New P6W Classification | Paper-Ready? |
|------------|----------------------|:---:|
| MOSEI main_awaf_slstm (inherited) | inherited_reference (P6T, 12ep) | ✅ |
| MOSEI fusion_gated (4ep) | formal_unfair_budget (4 vs 12ep) | ❌ |
| MOSEI awaf_no_interaction (4ep) | formal_unfair_budget | ❌ |
| MOSEI encoder_gru (4ep) | formal_unfair_budget | ❌ |
| MOSEI encoder_no_temporal (4ep) | formal_unfair_budget | ❌ |
| MOSEI fusion_mean (4ep) | formal_unfair_budget | ❌ |
| MOSI main_awaf_slstm (inherited) | inherited_reference (P6K, 20ep) | ✅ |
| MOSI all 8 variants (5ep, collapsed) | failed_collapse | ❌ |

## P6W Protocol

| Requirement | MOSI | MOSEI |
|-------------|------|-------|
| Code commit | P6W current | P6W current |
| Control re-run | 20 epochs | 12 epochs |
| All variants same budget as control | 20 epochs | 12 epochs |
| Seed | 42 | 42 |
| Fair comparison guaranteed | ✅ | ✅ |

Generated: 2026-06-20
