# MOSEI Data Chain Audit

## Status: BLOCKED

| Component | Status | Detail |
|-----------|:------:|--------|
| CSD files (7 files, ~30GB) | ✅ | `data/cmu_mosei_comp_seq/` |
| mmsdk import | ✅ | CMU-MultimodalSDK v0.0.6 |
| label.csv | ❌ | Not extracted |
| Feature directories | ❌ | Not extracted |
| TextFTMultimodalDataset support | 🟡 | Code ready, no data |
| train_textft_lora_mainline.py | 🟡 | dataset=mosei param supported |

## Required Files

```
data/mosei/label.csv                    # extracted from Labels CSD
data/features_strong_sequence_mosei_v3_T40/train/  # audio + vision features
data/features_strong_sequence_mosei_v3_T40/val/
data/features_strong_sequence_mosei_v3_T40/test/
```

## Action

CSD → label.csv + feature extraction needed before MOSEI training can start.
