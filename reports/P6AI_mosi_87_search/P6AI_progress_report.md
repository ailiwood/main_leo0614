# P6AI Progress Report — MOSI 87+ Search

**Date**: 2026-06-21  
**Branch**: `p6ai-mosi-87-performance-recovery`  
**Target**: ACC2_Non0 > 87.00% on MOSI official test

---

## Current Status: 84.60% — Best TAV result, but below 87% target

| Attempt | Architecture | ACC2 | Corr | Val ACC2 | Status |
|---------|-------------|------|------|----------|--------|
| P6K ref | T+A residual (v3_T40) | 88.72% | 0.851 | — | Different features, not TAV |
| P6AD F0 | Canonical AWAF | 42.23% | -0.03 | — | Softmax collapse |
| P6AF RtA | Canonical AWAF | 44.21% | -0.11 | — | Softmax collapse |
| P6AG T1 | Text-Anchored V1 | 82.01% | 0.749 | 87.50% | Working baseline |
| Family A | T+A residual (retry) | 46.04% | 0.145 | — | Gate collapsed to 1.0 |
| **Family B** | **Text-Anchored V2** | **84.60%** | **0.795** | **87.96%** | **Best TAV so far** |

## Family B Details

- Learnable gate bias: stuck at -2.0 (no learning)
- Bounded alpha: stuck at 0.5 (no learning)
- r_a: 0.374 ± 0.098 (per-sample variation)
- r_v: 0.544 ± 0.065 (per-sample variation)
- Text anchor alone: 84.76% — very strong
- TAV final: 84.60% — audio/vision slightly degrade
- Best val ACC2: 87.96% — shows capacity exists

## Remaining Gap to 87%

- Gap: 2.4pp (84.60% → 87.00%)
- P6K T+A achieved 88.72% — demonstrates 87+ is possible on MOSI
- Key challenge: audio/vision don't add net positive value despite correct architecture
- Val-test gap (87.96% → 84.60% = 3.36pp) — overfitting on small dataset

## Remaining Families

| Family | Description | Status |
|--------|-------------|--------|
| C | Text-Query Cross-Modal Adapter | Not started |
| D | TA-to-TAV Curriculum | Not started (needs working TA first) |

## Next Steps

1. Family C: Text-query attention to select relevant audio/vision frames
2. Family D: Curriculum learning (TA first, then TAV)
3. Both require Families A/B to establish baselines (done)
