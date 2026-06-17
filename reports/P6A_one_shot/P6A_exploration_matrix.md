# P6A Exploration Matrix

## Priority Ranking

| # | Experiment | Feature | Model | Strategy | Seed | Epochs | Est. Time | Threshold | Risk |
|---|-----------|---------|-------|----------|------|--------|-----------|-----------|------|
| 1 | Vision-L/14 extraction | MP4→CLIP-L/14 | — | — | — | — | ~30min | 100% success | MP4 decode errors |
| 2 | Vision-L/14 V2 | vis_l14 | V2 | one-stage | 42 | 60 | ~18min | >82.47% | Vision noise |
| 3 | Text upper bound | current text | text-only | hparam search | 42 | 30 | ~45min | Find best text config | None |
| 4 | Best text + V2 | current | V2+best_text | one-stage | 42 | 60 | ~18min | >82.47% | Overfit |
| 5 | Vision-L/14 V2 s2024 | vis_l14 | V2 | one-stage | 2024 | 60 | ~18min | >82% | Only if #2 succeeds |
| 6 | Two-stage best | best feat | V2 | two-stage | 42 | 60 | ~25min | >one-stage | Degradation |
| 7 | WeakNeg reweight | best feat | V2 | reweight | 42 | 60 | ~18min | WeakNeg improved | Overall drop |
| 8 | Engineering upper bound | best | ensemble | calibration | — | — | ~5min | >85% | Not main model |
| 9 | MOSEI SDK repair | — | — | SDK fix | — | — | ~15min | Import works | May fail |
| 10 | MLCL/CASP audit | — | — | literature | — | — | ~10min | Verify citations | No local code |
