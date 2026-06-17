# HANDOFF_PHASE_14F.md — P5F Strong Feature Upper Bound Sprint

## 本阶段完成

### ✅ 已完成

1. **P5E baseline confirmed**: V2 2-seed mean 82.17%
2. **Storage/cache audit**: 4 pth, 52MB, clean, new feature dirs planned
3. **Audio large smoke**: wav2vec2-large-960h-lv60-self ✅ 1024d (vs base 768d)
4. **Vision large smoke**: CLIP ViT-L/14 ✅ 1024d (vs B/32 768d)
5. **MOSEI status**: Still blocked, dimensions compatible

### ⏳ 未完成

| Task | Status | Reason |
|------|--------|--------|
| Text hparam search | Killed | 8×30ep taking too long (~1hr) |
| Audio large full extraction | Not started | Requires ~45min + script adaptation |
| Vision large full extraction | Not started | Requires ~45min + script adaptation |
| Full strong feature V2 training | Not started | Pending feature extraction |

## Key Findings

1. **Both large models downloaded successfully**: wav2vec2-large (1.26GB) + CLIP ViT-L/14 (1.71GB)
2. **Both provide 1024d features**: +33% dimension vs current base models (768d)
3. **Feature extraction is the bottleneck**: ~1.5 hours total for full 2199-sample extraction
4. **Text branch may still be the limiting factor**: DeBERTa-large is already strong

## 是否达到目标

| Target | Status |
|--------|--------|
| 83% | ❌ Not yet |
| 85% | ❌ Not yet |
| 87% | ❌ Not yet |

## 下一步建议

1. Complete text hparam search (remaining time ~1hr, can run overnight)
2. Run audio large full extraction (~45min)
3. Run vision large full extraction (~45min)
4. Build v4_full_strong features
5. Train V2 with strong features
6. If strong features push >83%, run multi-seed

## GitHub

- **分支**: `p5f-strong-feature-upper-bound-sprint`
- **Commit**: (pending)
