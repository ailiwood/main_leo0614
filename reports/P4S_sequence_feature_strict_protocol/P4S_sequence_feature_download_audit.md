# P4S Sequence Feature Download Audit

## Source
- URL: https://drive.google.com/uc?id=12HbavGOtoVCqicvSYWl3zImli5Jz0Nou
- From: MLCL (YetZzzzzz/MLCL) README, MIT license
- File: mosi_mcl.pkl (14.2MB pickle)

## Feature Format

| Property | Value |
|----------|-------|
| Format | Python pickle protocol 3 |
| Splits | train(1281), dev(229), test(685) |
| Per-sample | (features_tuple, label, id) |
| Text | list of word strings, T~12 |
| Audio | (T, 47) COVAREP float32 |
| Vision | (T, 74) FACET float32 |
| Alignment | Aligned to word-level |
| Labels | [-3.0, +3.0] |

## Sequence vs TMDC-v1 Comparison

| Property | MLCL Sequence | TMDC-v1 (Old) |
|----------|:--:|:--:|
| T (sequence length) | ~12 | **1** |
| Text dim | (word list → 300d embed) | 1024 (DeBERTa pool) |
| Audio dim | 47 (COVAREP) | 1024 (wav2vec pool) |
| Vision dim | 74 (FACET) | 1024 (CLIP pool) |
| sLSTM temporal | ✅ Real T>1 | ❌ T=1 degraded |
