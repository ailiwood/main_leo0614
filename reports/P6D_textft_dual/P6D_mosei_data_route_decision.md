# P6D MOSEI Data Route Decision

## Finding: MOSEI raw text data IS available ✅

| Item | Status |
|------|--------|
| Raw text | ✅ CSV with `text` column, all samples |
| Labels | ✅ sentiment [-3, 3] |
| Train/Val/Test | 16274 / 1861 / 4653 ✅ |
| Label format | CSV: video, start_time, end_time, sentiment, text, ASR |

## SDK Status

❌ mmsdk install blocked by PyPI mirror SSL issues + setuptools 81 conflict.
Cannot install via pip or GitHub.

## Data Route Decision

**Route B (standard data)**: MOSEI raw text and labels are available without SDK.
Text-only RoBERTa-large fine-tuning is feasible immediately.

- Audio: WAV chunks available (Train_modified/, Val_modified/, Test_modified/) but no pre-extracted features
- Vision: Not available
- Text: ✅ Ready for RoBERTa-large fine-tune

## Current Action

RoBERTa-large text-only MOSEI training launched (10 epochs, 16274 samples).
This will establish the text baseline for MOSEI.
