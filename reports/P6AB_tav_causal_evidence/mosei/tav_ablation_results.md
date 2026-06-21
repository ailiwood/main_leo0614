# P6AB MOSEI TAV Ablation Results

**Date**: 2026-06-21
**Protocol**: 4 epochs, seed=42, cohort=mosei_official_tav_intersection_v1

## Results Table

| Var | Description | Mods | Best Ep | ACC2 | F1 | MAE | Corr | ACC7 | Params(M) | w_t | w_a | w_v |
|-----|-------------|------|---------|------|----|-----|------|------|-----------|-----|-----|-----|
| F0 | Full TAV AWAF sLSTM | T+A+V | 4 | 78.32 | 83.30 | 0.7345 | 0.5622 | 43.31 | 4.84 | 0.435 | 0.263 | 0.302 |
| F1 | No Vision (T+A) | T+A | 4 | 76.65 | 82.53 | 0.7421 | 0.5532 | 41.41 | 4.10 | 0.368 | 0.301 | 0.331 |
| F2 | No Audio (T+V) | T+V | 4 | 78.72 | 82.94 | 0.7155 | 0.5819 | 42.62 | 4.26 | 0.341 | 0.365 | 0.294 |
| F3 | Global Static Weights | T+A+V | 4 | 79.30 | 83.70 | 0.7064 | 0.5972 | 43.40 | 3.59 | 0.333 | 0.333 | 0.333 |
| F4 | Fixed Mean (1/3 each) | T+A+V | 4 | 79.30 | 83.70 | 0.7064 | 0.5972 | 43.40 | 3.59 | 0.333 | 0.333 | 0.333 |
| F5 | AWAF No Interaction | T+A+V | 4 | 75.96 | 82.15 | 0.7489 | 0.5540 | 42.08 | 4.45 | 0.365 | 0.255 | 0.380 |
| E1 | LSTM Temporal Encoder | T+A+V | 4 | 77.78 | 81.86 | 0.7096 | 0.5891 | 43.64 | 4.97 | 0.482 | 0.245 | 0.273 |

## Key Comparisons

| Comparison | What it measures | F0 ACC2 | Variant ACC2 | Delta |
|------------|-----------------|---------|-------------|-------|
| F0 vs F1 | Vision modality contribution | 78.32 | 76.65 | +1.67 |
| F0 vs F2 | Audio modality contribution | 78.32 | 78.72 | -0.39 |
| F0 vs F3 | Sample-level dynamic weight contribution | 78.32 | 79.30 | -0.97 |
| F0 vs F4 | Dynamic/learned fusion contribution | 78.32 | 79.30 | -0.97 |
| F0 vs F5 | Hadamard interaction term contribution | 78.32 | 75.96 | +2.37 |
| F0 vs E1 | sLSTM vs plain LSTM contribution | 78.32 | 77.78 | +0.55 |
