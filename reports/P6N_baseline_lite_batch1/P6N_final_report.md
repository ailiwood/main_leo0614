# P6N Final Report

## Branch/Commit

`p6n-baseline-lite-batch1` / based on P6M `81fb849`

## P6M Config Sync

All 6 missing config files updated with baseline-lite policy block.

## Models Implemented

| Model | File | Params | Forward | Backward | 1ep Smoke |
|-------|------|:------:|:-------:|:--------:|:---------:|
| TFN-lite | `models/baselines/tfn_lite.py` | 1.83M | ✅ | ✅ | ✅ |
| LMF-lite | `models/baselines/lmf_lite.py` | 1.75M | ✅ | ✅ | ✅ |
| MulT-lite | `models/baselines/mult_lite.py` | 1.72M | ✅ | ✅ | ✅ |
| SelfMM-lite | `models/baselines/self_mm_lite.py` | 1.71M | ✅ | ✅ | ✅ |

## Simplified Differences vs Original Papers

| Model | Simplification |
|-------|---------------|
| TFN-lite | Factorized tensor when H>32; GRU not LSTM; text embedding not RoBERTa |
| LMF-lite | GRU encoder; no pretrained features |
| MulT-lite | Text-centric only (2 of 6 cross-modal directions); GRU not transformer encoder |
| SelfMM-lite | Real label auxiliary supervision (not self-supervised pseudo-label) |

## Smoke Results (1 epoch, 20 batches)

| Model | test_ACC2 | val_ACC2 | MAE |
|-------|:---------:|:--------:|:---:|
| TFN-lite | 42.53% | 57.41% | 1.42 |
| LMF-lite | 42.23% | 57.41% | 1.43 |
| MulT-lite | 42.23% | 57.41% | 1.57 |
| SelfMM-lite | 57.77% | 42.59% | 1.41 |

Expected for 1-epoch baseline-lite (no RoBERTa, lightweight GRU encoder).

## New/Modified Files

- `models/baselines/base_baseline.py` (rewritten)
- `models/baselines/tfn_lite.py` (new)
- `models/baselines/lmf_lite.py` (new)
- `models/baselines/mult_lite.py` (new)
- `models/baselines/self_mm_lite.py` (new)
- `scripts/train_baseline_lite.py` (new)
- `configs/experiments/p6n_baseline_lite_batch1/` (4 configs)
- `reports/P6N_baseline_lite_batch1/` (3 reports)
- 6 project config files updated with baseline-lite policy

## Next Steps

1. P6O: 10-epoch quick run for batch-1 models
2. P6O: Implement batch-2 (MISA/MMIM/MLCL/DLF-lite)
3. After quick runs: seed42 full training for best performers
