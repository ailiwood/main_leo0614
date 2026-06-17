# P5D .gitignore Audit

> Date: 2026-06-17
> File: `.gitignore` (101 lines)

## Coverage Checklist

| Pattern | Covered? | Lines |
|---------|----------|-------|
| `outputs/` | ✅ | 31 |
| `checkpoints/` | ✅ | 45 |
| `data/raw/` | ✅ | 38 |
| `data/features*/` | ✅ | 39 |
| `data/mosi/` | ✅ | 41 |
| `data/CMU-MOSEI/` | ✅ | 42 |
| `*.pth` | ✅ | 52 |
| `*.pt` | ✅ | 51 |
| `*.ckpt` | ✅ | 53 |
| `*.bin` | ✅ | 54 |
| `*.csd` | ✅ | 48 (added 06171755) |
| `*.h5` | ✅ | 56 |
| `*.hdf5` | ✅ | 47 (added 06171755) |
| `*.npz` | ✅ | 61 |
| `*.npy` | ✅ | 60 |
| `*.pkl` | ✅ | 62 |
| `*.wav` | ✅ | 68 |
| `*.mp4` | ✅ | 70 |
| `*.zip` | ✅ | 78 |
| `__pycache__/` | ✅ | 2 |
| `archives/` | ✅ | 95 |
| `external/` | ✅ | 92 |
| `experiments/` | ✅ | 89 |
| `env/` | ✅ | 13 |

## Changes Made

Added two missing patterns:
- `*.hdf5` (line 47)
- `*.csd` (line 48)

## Safe Patterns

Note: `data/` is NOT globally excluded — only specific subdirectories.
This is intentional to allow `data/*.py` source files to be tracked.

`*.zip` is excluded — review package zip will NOT be committed to git.
This is correct: zip is for local upload to Web AI only.

## Verdict

✅ .gitignore is comprehensive and correct.
✅ All data/checkpoint/feature/raw patterns covered.
