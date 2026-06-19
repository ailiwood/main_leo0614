# MOSEI Dataset Verification

## Full MOSEI Confirmed

| Item | Value |
|------|-------|
| Source | CMU-MOSEI official distribution (2023) |
| Path | `data/CMU-MOSEI/CMU-MOSEI-20230514T151450Z-001/CMU-MOSEI/Labels/` |
| Train | 16,274 (official) |
| Valid | 1,861 (official) |
| Test | 4,653 (official) |
| **Total** | **22,788** |
| Split source | official ✅ |
| Paper usable | ✅ |

## Modalities Available

| Modality | Source | Dim | Status |
|----------|--------|:---:|:------:|
| Text (raw) | Data_*.csv 'text' column | RoBERTa tokenize | ✅ Ready |
| Text (GloVe) | CSD TimestampedWordVectors | 300d | 🟡 Needs adapter |
| Audio | CSD COVAREP | 74d | 🟡 Needs CSD alignment |
| Vision | CSD OpenFace2 | 713d | 🟡 Needs CSD alignment |
| Labels | Data_*.csv 'sentiment' | 1 (regression) | ✅ Ready |

## Blocker: CSD-to-CSV Alignment

CSD segment IDs (video_id) don't match CSV segment IDs (video_starttime). Audio/vision features need alignment before text_audio mode is possible.

## Current Status

- text_only: MOSEI-ready (but too slow for full training)
- text_audio: BLOCKED pending CSD-CSV audio alignment
