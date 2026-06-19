# P6L MOSEI Vision Candidate Plan

## Key Constraints

1. **MOSI 已退出 vision 主路径** — MOSI text_audio (T+A) = mainline
2. **MOSEI 不得提前放弃 vision** — MOSEI text_audio 结果出来前不预设结论
3. **MOSI vision 结论不向外推** — 不同数据集，不同模态贡献

## MOSEI Vision Validation Plan

### Phase 1: MOSEI text_audio baseline (P6L)
- Run MOSEI text_audio s42 + s2024
- Measure text_base ACC2, identify performance baseline

### Phase 2: MOSEI vision signal check (P6L+)
- Run MOSEI vision_only (like MOSI 65.28% check)
- Run MOSEI av_only
- Determine if MOSEI vision signal > MOSI vision signal

### Phase 3: Decision gate
- If MOSEI text_audio > MOSI text_audio → T+A may be sufficient for MOSEI too
- If MOSEI vision signal > 70% → consider T+A+V
- If MOSEI text_audio < target → evaluate all modality combinations

## Current Blocker

MOSEI features not extracted from CSD files. Need:
1. Extract labels → `data/mosei/label.csv`
2. Extract audio features → `data/mosei/features/`
3. Extract vision features → `data/mosei/features/`
4. Create `TextFTMultimodalDataset` compatible structure

CSD data (7 files, ~30GB) at `data/cmu_mosei_comp_seq/` is confirmed loadable via mmsdk.
