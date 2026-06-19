# P6K Stage Decision

## Questions & Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Class-based textbase recovered to 86%+? | ✅ **Yes** — 86.13% (within 0.3% of P6H 86.43%) |
| 2 | text_audio conservative stable? | ✅ **Yes** — s42=88.72%, s2024=86.89%, mean=87.8% |
| 3 | text_audio exceeds text_only? | ✅ **Yes** — 88.72% > 83.99% (+4.73%) |
| 4 | MOSI candidate main model? | ✅ **text_audio conservative** (T+A, no residual claim) |
| 5 | Allow MOSEI conservative? | 🟡 **Conditional** — MOSI results are strong, but residual_gain still 0 |
| 6 | Baseline smoke completed? | 🟡 **Partial** — MLCL code ready; MMSA (6 baselines) needs cloning |
| 7 | Allow baseline full training? | ❌ **Not yet** — needs MMSA setup + smoke verification first |
| 8 | Still block Ch5 conclusions? | ✅ **Yes** |

## Decision: Case A → Text-Audio Conservative Mainline

- textbase recovered ✅
- text_audio multi-seed stable (87.8% mean) ✅
- text_audio exceeds text_only ✅

## Action Items

1. ✅ text_audio conservative = MOSI candidate main model
2. 🟡 MOSEI text_audio conservative pending (user decision)
3. 🟡 Clone MMSA for classic baseline smoke
4. 🟡 Run MLCL 3ep smoke
5. ❌ No MOSEI long training yet
6. ❌ No baseline full training yet
