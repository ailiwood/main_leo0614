# HANDOFF_PHASE_04A.md

## 环境
mme_xlstm_stable: PyTorch 2.11.0+cu128 STABLE, CUDA 12.8, RTX 5070 Ti

## 代码冻结检查
7/7 checks passed ✅ — C0-only, AWAF full, sLSTM correct, metrics complete

## AWAF 正则化裁决

| Config | ACC2_NZ | vs base | Decision |
|--------|:--:|:--:|:--:|
| base (md=0.1) | 73.63% | — | baseline |
| entropy 0.01 | 74.24% | +0.61% | helps slightly |
| entropy 0.05 | 73.40% | -0.23% | hurts |
| **stronger_md (md=0.2)** | **75.61%** | **+1.98%** | **✅ ADOPTED** |
| entropy + md_0.2 | 74.09% | +0.46% | no synergy |

**Decision: md_prob=0.2, no entropy reg.**

## MOSI 3-Seed 60-Epoch Results

| Seed | ACC2_NZ | MAE | Corr | F1_NZ | Best Ep |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 42 | 75.00% | 1.009 | 0.584 | 68.58 | 20 |
| 2024 | 75.30% | 0.992 | 0.601 | 70.55 | 26 |
| 1234 | 75.46% | 0.999 | 0.592 | 69.33 | 21 |
| **Mean±Std** | **75.25±0.23%** | **1.000±0.008** | **0.592±0.009** | | |

## AWAF 权重

w_t=0.47-0.68, w_v=0.26-0.40 — balanced, no collapse ✅
Stronger md_prob effectively prevents text collapse

## MOSEI Readiness
- Text ✅, Audio ✅, Vision ❌
- Need user to provide video/frame source before P4B

## Next Steps
- P4B: MOSEI training (after vision features obtained)
- Or P5/P7: MOSI ablation experiments

## Confidence
P4A MOSI results: confidence B, paper_usable=pending (waiting for MOSEI + sequence features)
