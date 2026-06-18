# P6E MOSEI SDK Install Report

## Install Success

| Method | Package | Result |
|--------|---------|--------|
| PyPI | `cmu-multimodal-sdk` v0.0.6 | ✅ Installed |
| Import | `from mmsdk import mmdatasdk` | ✅ OK |

## Computational Sequence Access

| Attempt | URL | Result |
|---------|-----|--------|
| High-level features | cmu_mosei.highlevel | ❌ HTTP 404 |
| Labels | cmu_mosei.labels | ❌ HTTP 404 |

CMU immortal server (`immortal.multicomp.cs.cmu.edu`) returns 404 for .csd files.

## Fallback: CSV Data

MOSEI raw text + labels in `data/CMU-MOSEI/.../Labels/*.csv` are usable:
- Sentiment labels [-3,3] ✅
- Raw text (transcripts) ✅
- Audio WAV chunks available
- Vision features NOT available (no .csd, no video files)

## Recommendation

- Use CSV for labels + raw text (already validated: 88.13% text-only)
- Audio features need extraction from WAV chunks
- Vision features blocked without raw video
