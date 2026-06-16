# P4T File Cleanup Classification

## Category A: Core Code (KEEP in Git)

```
models/encoders/slstm.py
models/fusion/awaf.py
models/heads.py
models/ours_xlstm_fusion.py
models/enhancements/deconv.py, cme.py, cme_residual.py (attempted)
engine/strict_trainer.py
engine/trainer.py
data/dataset.py (TMDC-v1)
data/sequence_dataset.py (MLCL standard)
utils/metrics.py
utils/seed.py
configs/ (all .yaml)
scripts/ (all active .py)
```

## Category B: Research Records (KEEP in Git)

```
HANDOFF_PHASE_*.md
memory.md
docs/DECISIONS.md, HYPERPARAMS.md
reports/ (key summaries)
README_PROJECT_STATUS.md
```

## Category C: Archive (NOT in Git, move to archives/)

```
outputs/ (all P0-P4S run outputs)
experiments/ (v9 weights)
results_20260307/
tmdc_adapter/hf_cache/ (~15GB)
external/MLCL/ (~100MB)
external/features_mlcl/ (14MB)
logs/
```

## Category D: Git-Ignore But Keep Local

Already configured in .gitignore:
outputs/, data/features*/, external/, *.pth, *.npy, *.pkl, *.wav

## Category E: Delete Candidates (User MUST confirm)

```
~$0论文初稿03181845.docx (Word temp)
**pycache**/
.pytest_cache/
旧重复日志
```

**NO FILES DELETED YET — awaiting user confirmation.**
