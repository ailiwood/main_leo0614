# P6J Stage Decision

## Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Latest code reproduce P6H 86.43%? | ❌ No — P6J textbase=83.99%. P6H original confirmed: 86.43% |
| 2 | Text_conf delta direction correct? | ❌ No — sign_correct=48.69% (≈random) |
| 3 | Residual net correction benefit? | ❌ No — net_correct_gain=-10 (harmful) |
| 4 | Best A/V combo? | text_audio (88.72% test, but gain=0.00%) |
| 5 | Continue strong residual? | ❌ No — all modes gain=0, delta direction random |
| 6 | Switch to text-dominant conservative? | ✅ Yes |
| 7 | Allow MOSEI? | ❌ No |
| 8 | Allow baseline smoke? | 🟡 Conditional (only after textbase fixed) |
| 9 | Still block Ch5 conclusions? | ✅ Yes |

## Decision: Case B → Text-Dominant Conservative

- P6H inline code: text_base=86.43% ✅ (reproducible)
- P6I/P6J class-based code: text_base=83.99% 🔴 (regression)
- Residual: delta direction random, all modes gain=0.00%
- Decision: **Stop strong residual, fix textbase regression, adopt text-dominant approach**

## Key Evidence

1. **Residual delta direction = RANDOM** (sign_correct ≈ 45-50% in all 4 modes)
2. **A/V signal too weak** for directional correction (audio=56.9%, vision=65.3%)
3. **Text_audio reaches 88.72%** — best configuration, but from text_base co-training, not residual
4. **Code regression exists**: inline→class-based loses 2.44% ACC2

## Next Steps

1. Fix textbase regression in class-based code (restore 86.43%)
2. Retain text+audio co-training as main configuration
3. Prepare for MOSEI with text-only or text+audio (no residual)
4. Allow baseline smoke after textbase fixed
