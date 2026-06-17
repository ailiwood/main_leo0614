# P6A State Reading Report

## Files Read

| File | Status |
|------|--------|
| HANDOFF_PHASE_15G.md | MISSING (P5G last commit had no handoff file) |
| HANDOFF_PHASE_14F.md | EXISTS ✅ |
| HANDOFF_PHASE_13E.md | EXISTS ✅ |
| memory.md | EXISTS ✅ |
| docs/DECISIONS.md | EXISTS ✅ |
| models/deeptext_xlstm_awaf_residual_v2.py | EXISTS ✅ |
| engine/residual_losses_v2.py | EXISTS ✅ |
| data/strong_sequence_dataset.py | EXISTS ✅ |
| scripts/train_deeptext_xlstm_awaf_residual_v2.py | EXISTS ✅ |
| configs/models/deeptext_xlstm_awaf_residual_v2_mosi.yaml | EXISTS ✅ |

## Current State Summary

- **Best model**: P5E V2, 2-seed mean 82.17% (s42: 82.47%, s2024: 81.86%)
- **Audio large**: wav2vec2-large degrades (-2.29%), marked FAILED
- **Vision large**: Not yet extracted from MP4
- **MOSEI**: SDK blocked
- **Current features**: text=1024 (DeBERTa), audio=768 (wav2vec2-base), vision=768 (CLIP-B/32)
- **Raw MP4**: Present at data/mosi/Raw/<video_id>/<seg>.mp4
