# P5F Audio Large Smoke Test

## Model: facebook/wav2vec2-large-960h-lv60-self

| Item | Result |
|------|--------|
| Downloaded | ✅ 1.26GB |
| Hidden size | **1024** (vs base 768, +33%) |
| Loading | ✅ Successful |
| Audio path | data/mosi/wav/<id>/*.wav (segmented utterances) |
| Current WAV loading | ⚠️ Need to adapt extraction script for segmented WAVs |

## Recommendation

✅ Proceed with full extraction if time allows (~45 min for 2199 samples)
⚠️ Need to adapt extraction to handle segmented utterance WAV files
⚠️ Feature format: 1024d frames (vs current 768d)
