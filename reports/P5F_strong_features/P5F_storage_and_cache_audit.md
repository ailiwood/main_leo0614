# P5F Storage & Cache Audit

## Current Space

| Directory | Size | Notes |
|-----------|------|-------|
| outputs/ | 52MB | 4 pth files only (cleaned) |
| data/features_strong_sequence_mosi_v3_T40/ | ~500MB | Current active features |
| data/features_strong_sequence_mosi_v3_T32/ | ~400MB | Previous version |
| data/features_strong_sequence_mosi_v2/ | ~400MB | Older version |
| data/features_strong_sequence_mosi/ | ~300MB | Old version |
| data/CMU-MOSEI/ | ? | Raw audio (gitignored) |
| data/mosi/ | ? | Raw MP4/WAV (gitignored) |

## HF Cache

Location: `~/.cache/huggingface/`
Space: Not checked (will use existing cache)

## New Feature Directories (planned)

- `data/features_strong_sequence_mosi_v3_audio_large/` — wav2vec2-large (1024d)
- `data/features_strong_sequence_mosi_v3_vision_large/` — CLIP ViT-L/14 (768d or 1024d)
- `data/features_strong_sequence_mosi_v4_full_strong/` — combined best features

## Policy

- Save only best_model.pth
- No epoch checkpoints
- No feature files committed to git
- Clean old features after P5F if successful
