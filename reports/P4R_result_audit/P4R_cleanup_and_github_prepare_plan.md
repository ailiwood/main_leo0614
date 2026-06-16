# P4R GitHub Upload Preparation Plan (DRAFT ONLY — NO DELETION)

## Keep (Core Code)

```
models/encoders/slstm.py
models/fusion/awaf.py
models/heads.py
models/ours_xlstm_fusion.py
models/enhancements/deconv.py, cme.py, cme_residual.py
utils/metrics.py, seed.py
engine/trainer.py
data/dataset.py
configs/default.yaml, models/ours_c0_p4a_final.yaml
scripts/smoke_forward.py, train_main.py
claude.md, workflow.md, memory.md, HANDOFF*.md
```

## Archive

```
experiments/v9_roberta*/       # V9 weights ~115MB
outputs/P2_smoke/, P3A_*/      # Old experiment outputs
results_20260307/               # 1st round results
tmdc_adapter/hf_cache/          # ~10GB HuggingFace cache
0613实验FINAL_REPORT.md          # Old V9 report
```

## Candidates for Deletion (user confirm first)

```
~$0论文初稿03181845.docx        # Word temp file
logs/*.log                      # Old training logs
.pth files in experiments/      # Large weight files
```

## .gitignore Suggestions

```
outputs/
experiments/
logs/
*.pth
*.pkl
*.npy
env/
__pycache__/
.ipynb_checkpoints/
tmdc_adapter/hf_cache/
~$*
```

## Recommended GitHub Upload

Minimal code set (~50 files, <1MB excluding docx):
- All .py files in models/, utils/, engine/, data/, scripts/, configs/
- All .md files
- Thesis .docx
- .gitignore
