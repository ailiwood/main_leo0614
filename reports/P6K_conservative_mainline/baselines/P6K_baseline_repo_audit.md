# P6K Baseline Repository Audit

## Status Summary

| Baseline | Code Available | Features Available | Smoke Ready | Blocked Reason |
|----------|:---:|:---:|:---:|---|
| TFN | ❌ | ❌ | ❌ | Requires MMSA framework |
| LMF | ❌ | ❌ | ❌ | Requires MMSA framework |
| MulT | ❌ | ❌ | ❌ | Requires MMSA framework |
| MISA | ❌ | ❌ | ❌ | Requires MMSA framework |
| Self-MM | ❌ | ❌ | ❌ | Requires MMSA framework |
| MMIM | ❌ | ❌ | ❌ | Requires MMSA framework |
| MLCL | ✅ | ✅ (mosi_mcl.pkl) | 🟡 | Needs env setup + data adaptation |
| CASP | ❌ | ❌ | ❌ | TTA method; requires separate env |

## MMSA Framework (TFN/LMF/MulT/MISA/Self-MM/MMIM)

- Repo: https://github.com/thuiar/MMSA
- Status: **Not cloned**. All 6 classic baselines depend on MMSA.
- Action: Clone MMSA, install dependencies, adapt MOSI data format.
- Estimate: 2-4 hours for setup + 6 × 0.5h per baseline smoke = 3-5h total.

## MLCL

- Repo: https://github.com/YetZzzzzz/MLCL
- Local: `external/MLCL/`
- Features: `external/features_mlcl/mosi_mcl.pkl`
- Status: Code and features present.
- Action: Install env, adapt data loader for MOSI split, run 3ep smoke.
- Estimate: 1-2 hours.

## CASP

- Repo: https://github.com/zrguo/CASP
- Status: **Not cloned**. TTA method, requires separate environment.
- Note: Must label as TTA in all tables.

## Recommendation

1. Clone MMSA for classic baselines (highest priority: Self-MM, MulT)
2. Set up MLCL env and run 3ep smoke
3. Defer CASP to later phase
