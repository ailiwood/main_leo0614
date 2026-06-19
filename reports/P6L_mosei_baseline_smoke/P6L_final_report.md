# P6L Final Report

## 1. Branch/Commit/Env

| Item | Value |
|---|---|
| Branch | `p6l-mosei-baseline-smoke` |
| Base commit | `59b216f` (P6K) |
| Env | `mme_xlstm_stable` |

## 2. MOSI Mainline Lock

| Decision | Status |
|---|---|
| MOSI mainline = text_audio conservative (T+A) | ✅ Written to DECISIONS.md (D039) |
| Vision exits MOSI main path | ✅ D039 |
| Residual/delta not claimed as primary gain | ✅ D040 |
| MOSEI main model not locked yet | ✅ D042 |
| Ch5 still blocked | ✅ D043 |

## 3. 2025 Baseline Audit

| Model | Smoke Status | Can Enter Table | Blockers |
|-------|:-----------:|:---------------:|----------|
| MLCL | B | Yes | MOSI features ok; needs env + smoke run |
| DLF | B | Yes | MMSA data format adaptation needed |
| DPDF-LQ | B | Yes | Data download needed |

## 4. Classic Baseline (MMSA) Status

- MMSA cloned: `external/MMSA/` (commit `a94e65d`)
- 6 classic models supported: TFN, LMF, MulT, MISA, Self-MM, MMIM
- Smoke status: B — needs env setup + MOSI features in MMSA format

## 5. CASP Exclusion

CASP is TTA (test-time adaptation). Not mixed with end-to-end baselines. Separate table only.

## 6. Dataset Parameterization

| Test | Result |
|------|--------|
| MOSI backward compat | ✅ Pass (1284 samples) |
| MOSEI dataloader | ❌ Blocked (no label.csv / features) |

## 7. MOSEI Training

**Blocked**: MOSEI features must be extracted from CSD before training can start.
- 7 CSD files at `data/cmu_mosei_comp_seq/` (~30GB)
- Need feature extraction pipeline (labels + audio + vision)

## 8. MOSEI Vision Plan

Written at `mosei_vision_candidate_plan.md`. Decision deferred until MOSEI text_audio results available.

## 9. Credibility Assessment

| Result | Credibility | Can Enter Paper |
|--------|:----------:|:--------------:|
| P6K text_audio s42 (88.72%) | A | ✅ candidate |
| P6K text_audio s2024 (86.89%) | A | ✅ candidate |
| P6K textbase recovery (86.13%) | A | ✅ candidate |
| MLCL/DLF/DPDF-LQ | B (smoke only) | ❌ needs full run |
| MMSA classic | B (not run) | ❌ needs full run |
| MOSEI | N/A | ❌ blocked |

## 10. Next Steps

1. Extract MOSEI features from CSD
2. Run MOSEI text_audio s42 + s2024
3. Set up MLCL env + run smoke
4. Set up MMSA env + run Self-MM smoke
5. After MOSEI results: decide on vision inclusion
