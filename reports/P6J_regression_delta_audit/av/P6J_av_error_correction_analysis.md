# P6J A/V Error Correction Analysis

## Test Results (3 residual modes, old gate+delta)

| Mode | test_ACC2 | text_base | gain | sign_correct | err_reduced | TWFR | TRFW | net |
|---|---|---|---|---|---|---|---|---|
| text_audio | **88.72%** | 88.72% | 0.00% | 44.6% | 43.3% | 10 | 7 | **+3** |
| text_vision | 85.82% | 85.82% | 0.00% | 46.7% | 44.9% | 2 | 6 | **-4** |
| text_av | 86.13% | 86.13% | 0.00% | 49.7% | 48.8% | 3 | 2 | **+1** |
| text_conf | 85.06% | 85.06% | 0.00% | 48.7% | 43.6% | 8 | 18 | **-10** |

## Per-mode analysis

### text_audio
- **Best test ACC2: 88.72%** (highest of all modes!)
- net_correct_gain=+3 (marginal positive)
- Audio seems to help text_base learn better (perhaps through shared gradient)

### text_vision
- Lowest test ACC2: 85.82%
- net_correct_gain=-4 (negative)

### text_av
- Middle: 86.13%
- net_correct_gain=+1 (neutral)

### text_confidence
- Gate open (0.75) + large delta (0.23) = more harm
- net_correct_gain=-10 (most harmful!)

## Key Findings

1. **All modes: gain=0.00%** — residual never improves ACC2 on test
2. **sign_correct ≈ 44-50%** — delta direction is random in all modes
3. **text_audio reaches 88.72%** — highest text_base among all modes
4. **Vision-only is worst** — matching its weak signal (65.28%)
5. **text_confidence most harmful** — open gate amplifies wrong-direction delta

## Judgment

| Q | Answer |
|---|---|
| Which modality combo is most valuable? | **Audio** (88.72% test, best text_base) |
| Is audio just noise? | No — audio co-training improves text_base relative to text-only |
| Can vision stably correct weak_neg/near_zero? | No — sign_correct < 50% |
| Is AV better than single modality? | No — text_audio (88.72%) > text_av (86.13%) |

## Decision

- text_audio has net_correct_gain=+3 (marginal)
- But gain still 0.00% on ACC2 — residual not contributing to classification
- **Recommendation**: Stop strong residual, retain text+audio as best training configuration
