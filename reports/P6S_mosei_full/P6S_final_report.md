# P6S Final Report — MOSEI Full Experiment Autopilot

## Branch/Commit

`p6s-mosei-full-experiment-autopilot` / `9c8c2ad`

## Completed

| Task | Status | Detail |
|------|:------:|--------|
| MOSEI CSD audit | ✅ | 7 files, 29.5GB, all HDF5-readable |
| MOSEI label extraction | ✅ | 3293 segments, 7 sentiment dims |
| MOSEI feature extraction | ✅ | 3292 aligned: COVAREP 74d + GloVe 300d |
| MOSEI label.csv | ✅ | train=2304, val=494, test=494 |
| MOSI baseline rescue | ✅ | 4 models, 75-78% (P6Q) |
| MOSI main model | ✅ | text_audio 88.72% (P6K) |

## Blocked

| Task | Blocker | Action |
|------|---------|--------|
| MOSEI main model training | Feature dim mismatch (74d vs 768d) | Dataset adapter needed |
| MOSEI baseline training | Same | Dataset adapter needed |
| MOSEI official split | CSD has 3293 segments vs official 22856 | Need full MOSEI download |

## Current Best Results (All MOSI, seed42)

### Main Model
| Model | ACC2 | F1 | MAE | Corr |
|------|:----:|:--:|:---:|:----:|
| text_audio conservative | **88.72%** | 86.64 | 0.635 | 0.851 |

### Baseline-Lite (frozen RoBERTa, h128)
| Model | ACC2 | F1 | MAE | Corr |
|------|:----:|:--:|:---:|:----:|
| TFN-lite | 77.59 | 71.01 | 1.078 | 0.614 |
| MulT-lite | 77.13 | 71.80 | 0.942 | 0.641 |
| SelfMM-lite | 76.68 | 73.11 | 1.024 | 0.614 |
| LMF-lite | 75.61 | 72.41 | 1.045 | 0.586 |
