# P5D Git Push Report

> Date: 2026-06-17 17:55

## Push Details

| Item | Value |
|------|-------|
| Branch | `p5d-residual-stability-performance-sprint` |
| Commit | `35fafd6` |
| Remote | `origin` → `https://github.com/ailiwood/main_leo0614.git` |
| Push status | ✅ Already pushed (from P5D sprint commit) |
| GitHub branch | https://github.com/ailiwood/main_leo0614/tree/p5d-residual-stability-performance-sprint |
| GitHub PR | https://github.com/ailiwood/main_leo0614/pull/new/p5d-residual-stability-performance-sprint |

## Push Contents

- models/ (deeptext, encoders, fusion, pooling, interaction, modules, heads)
- engine/ (strict_trainer, losses)
- scripts/ (train, test, analyze, twostage)
- configs/ (mosi, mosei_sdk, default)
- reports/ (P5C, P5D, code_review)
- docs/ (DECISIONS, FINAL_MODEL_SPEC, 主模型大修决策)
- memory.md, CODE_REVIEW_ENTRYPOINT.md
- HANDOFF*.md

## Excluded (correctly gitignored)

- outputs/ (537MB, 34 pth files)
- data/features*/ (npz feature files)
- data/CMU-MOSEI/ (raw audio wav)
- data/raw/, data/mosi/
- archives/ (archive files)
- external/ (third-party code)
- *.pth, *.pt, *.ckpt, *.npz, *.npy, *.pkl, *.wav, *.mp4

## Remaining Commit

This code review handoff includes new files to be committed:
- reports/P5D_code_review_06171755/*
- HANDOFF_CODE_REVIEW_06171755.md
- .gitignore (updated with *.csd, *.hdf5)

Note: `review_packages/P5D_code_review_06171755.zip` is NOT committed (gitignored as *.zip).
This is intentional — the zip is for local upload to Web AI only.
