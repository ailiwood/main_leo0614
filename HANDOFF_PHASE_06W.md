# HANDOFF_PHASE_06W.md — P4W MOSI Performance Final

## Results: AWAF-Seq + Vision T>1 + Enhanced Training

| Metric | C0 (baseline) | P4V.1 AWAF-Seq | **P4W Enhanced** |
|--------|:--:|:--:|:--:|
| ACC2_NZ_reg | 72.7% | 73.9% | **78.4%** |
| MAE | 1.096 | 1.070 | **0.980** |
| Corr | 0.572 | 0.629 | **0.666** |
| F1_NZ_reg | 69.2 | 72.7 | **75.8** |
| Vision | T=1 | T=1 | **T=20** |

## What Changed

1. **Vision T>1** (CLIP-ViT per-frame, 20fps, 768d): +4.5% — the biggest lever
2. **Sign consistency loss** (weight=0.1): encourages reg/cls agreement
3. **Auxiliary unimodal heads** (weight=0.1): stabilizes training
4. **60 epochs + scheduler + AMP**: better convergence

## 4-Group Hparam Probe

| Config | ACC2 | Result |
|--------|:--:|--------|
| A_baseline (HD=256,CL=2,BS=32) | **78.8%** | ✅ BEST |
| B_larger_hd (HD=384) | 77.0% | Larger doesn't help |
| C_deeper_cr (CL=3) | 77.4% | Deeper cross-modal doesn't help |
| D_larger_bs (BS=64) | 76.2% | Larger batch worse |

## GitHub
- Branch: p4w-mosi-awafseq-performance
- Commit: 05c0ad5
- URL: https://github.com/ailiwood/main_leo0614/tree/p4w-mosi-awafseq-performance

## Next Steps for Web AI
1. 78.4% is close to 80% — can we reach it with tuning?
2. Approaching 85% likely requires better features or larger model
3. MOSEI readiness: features need to be built
4. Ablation design: which component contributes how much?
