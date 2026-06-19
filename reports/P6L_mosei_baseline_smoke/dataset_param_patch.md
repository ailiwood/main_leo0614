# P6L Dataset Parameterization Patch

## Changes

- `scripts/train_textft_lora_mainline.py`: `[DATA]` section now reads `data.dataset` from YAML
- Supports `dataset: "mosi"` or `dataset: "mosei"` in config
- Default: `mosi` if not specified
- Output paths include dataset name
- `csv_path` and `feature_root` configurable via YAML

## Smoke Results

| Test | Result |
|------|--------|
| MOSI backward compat | ✅ 1284 samples loaded |
| MOSEI dataloader | ❌ `data/mosei/label.csv` not found |

## MOSEI Blocker

MOSEI data is in CSD format at `data/cmu_mosei_comp_seq/` (7 files, ~30GB). Need to:
1. Extract labels to `data/mosei/label.csv`
2. Extract features to `data/features_strong_sequence_mosei_v3_T40/`
3. Use CMU-MultimodalSDK (`mmsdk/mmdatasdk`) for extraction
