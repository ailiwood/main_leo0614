# P5D Git Status Before Push

> Date: 2026-06-17 17:55
> Branch: p5d-residual-stability-performance-sprint

## Branch Info

- **Current branch**: `p5d-residual-stability-performance-sprint`
- **Base branch**: `p5c-deeptext-xlstm-awaf-residual-refactor`
- **Remote**: `origin` → `https://github.com/ailiwood/main_leo0614.git`

## Recent Commits

| SHA | Message |
|-----|---------|
| `35fafd6` | P5D improve residual stability and performance |
| `19d6e6c` | P5C refactor main model to DeepText xLSTM AWAF Residual |
| `bcf730c` | P5B handoff: text-only DeepMLP=80.2% > Full AWAF-Seq=78.8% |

## Working Tree Status

**Clean** — no uncommitted changes, no unstaged files.

## Large File Audit

| Check | Result |
|-------|--------|
| pth/pt/ckpt tracked? | ✅ None |
| outputs/ tracked? | ✅ None |
| data/raw/ tracked? | ✅ None (gitignored) |
| data/features*/ tracked? | ✅ None (gitignored) |
| MOSEI wav/mp4 tracked? | ✅ None (gitignored) |
| Untracked large files | ✅ Only in gitignored dirs (data/CMU-MOSEI/, data/features*/, outputs/) |
| *.csd tracked? | ✅ None |
| *.hdf5 tracked? | ✅ None |

## Tracked Data Files (safe)

Only `.py` source files and `README`:
- `data/README_data.md`
- `data/dataset.py`
- `data/sequence_dataset.py`
- `data/strong_sequence_dataset.py`

## Verdict

✅ Git state is clean and safe for code review push.
✅ No data leakage, no checkpoint bloat in git history.
