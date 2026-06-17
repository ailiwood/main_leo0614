# P6A Stage Decision

## Completed

| Task | Result |
|------|--------|
| State reading | P5E V2 baseline 82.17% confirmed |
| Exploration matrix | 10 experiments prioritized |
| Baseline metric audit | ACC2_Non0/F1_Non0 protocol documented |
| Vision-L/14 MP4 extraction | **2199/2199 success, 24min, 1024d** ✅ |
| MOSEI SDK repair | ❌ Still blocked (setuptools 81 conflict) |
| Baseline table draft | Classic + MLCL + CASP TTA structure |

## Pending (for P6B)

| Task | Priority | Reason |
|------|----------|--------|
| Vision-L/14 V2 training | #1 | Features ready, needs ~18min training |
| Text upper bound search | #2 | 12 configs × 30ep |
| V2 + vision_l14 seed2024 | #3 | If seed42 > 82.47% |
| Two-stage best candidate | #4 | Only if one-stage effective |
| WeakNeg reweight | #5 | Only if top-2 candidates |
| Engineering upper bound | #6 | Only if >83% |

## Threshold Status

| Target | Best Known | Gap |
|--------|-----------|-----|
| 83% | 82.47% (P5E V2) | -0.53% |
| 85% | 82.47% | -2.53% |
| 87% | 82.47% | -4.53% |

## Decision: Enter P6B

Vision-L/14 features are ready. P6B should:
1. Train V2 + vision_l14 seed42 (highest priority)
2. Run text upper bound search
3. Build full feature root with vision_l14
4. Run ablation, two-stage, weakneg as needed
