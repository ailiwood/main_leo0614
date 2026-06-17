# P5G Stage Decision

## Results

| Model | ACC2 | MAE | Corr |
|-------|------|-----|------|
| P5E V2 baseline (base features) | **82.47%** | 0.8111 | 0.7385 |
| P5G V2 audio_large (1024d) | 80.18% | 0.8197 | 0.7398 |

## Key Finding

**wav2vec2-large (1024d) underperforms wav2vec2-base (768d) by -2.29%.**

AWAF weight evidence: w_a dropped from 0.15 to 0.03 — model actively suppresses noisy audio features.

## Decision

1. Audio_large: ❌ NOT effective — degrades performance
2. Vision_large: Not tested (needs MP4 re-extraction pipeline)
3. Full_strong: Not applicable (audio_large is harmful)
4. **Current feature combination is already near-optimal for available models**

## Thresholds

| Target | Best | Status |
|--------|------|--------|
| 83% | 82.47% (P5E V2 s42) | ❌ -0.53% |
| 85% | — | ❌ |
| 87% | — | ❌ |

## Next Recommendation

**P5H**: Accept current features as ceiling, run formal multi-seed P5E V2 with ablation, prepare for paper.
