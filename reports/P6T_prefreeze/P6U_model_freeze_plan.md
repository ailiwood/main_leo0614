# P6T-PreFreeze: P6U Model Freeze Plan (Draft)

**Date**: 2026-06-19  
**Status**: Draft — pending MOSEI main model text_audio completion

## 1. Final Model List (proposed)

### Main Models (2)
| ID | Model | Dataset | Modality | Config | Code |
|----|-------|---------|----------|--------|------|
| M1 | TextFTLoRAXLSTMAWAFResidual | MOSI | text_audio | `configs/.../p6k_text_audio_conservative_s42.yaml` | `models/textft_lora_xlstm_awaf_residual.py` |
| M2 | TextFTLoRAXLSTMAWAFResidual | MOSEI | text_audio | `configs/.../p6t_prefreeze_mosei_mainline/mosei_main_text_audio_cached_s42.yaml` | same |

### Diagnostic (1)
| ID | Model | Dataset | Modality | Purpose |
|----|-------|---------|----------|---------|
| D1 | TextFTLoRAXLSTMAWAFResidual | MOSEI | text_only | Text-only baseline for multimodal gain |

### Baselines (11)
| Dataset | Models | Count |
|---------|--------|-------|
| MOSEI | MulT, SelfMM, TFN, LMF, MISA, MLCL, DLF | 7 |
| MOSI | MulT, SelfMM, TFN, LMF | 4 |

## 2. Excluded Models

| Model | Dataset | Reason |
|-------|---------|--------|
| MMIM-lite | MOSEI | Code identical to MISA-lite |
| TAV mainline | MOSEI | Vision features all-zero |
| P6N/P6O baselines | MOSI | Collapse (constant predictions) |
| P6S_repair4 baselines | MOSEI | Collapse (COVAREP -inf) |

## 3. Code Paths

| Component | Path |
|-----------|------|
| Main model | `models/textft_lora_xlstm_awaf_residual.py` |
| Baseline base | `models/baselines/base_baseline.py` |
| MulT-lite | `models/baselines/mult_lite.py` |
| SelfMM-lite | `models/baselines/self_mm_lite.py` |
| TFN-lite | `models/baselines/tfn_lite.py` |
| LMF-lite | `models/baselines/lmf_lite.py` |
| MISA-lite | `models/baselines/misa_lite.py` |
| MLCL-lite | `models/baselines/mlcl_lite.py` |
| DLF-lite | `models/baselines/dlf_lite.py` |
| AWAF | `models/fusion/awaf.py` |
| sLSTM | `models/encoders/slstm.py` |
| Metrics | `utils/metrics.py` |
| Data loading | `data/textft_multimodal_dataset.py` |

## 4. Config Paths (proposed P6U freeze/)

```
configs/experiments/p6u_freeze/
├── mosi/
│   ├── main_text_audio_s42.yaml
│   └── main_text_only_s42.yaml (optional)
├── mosei/
│   ├── main_text_audio_s42.yaml
│   ├── main_text_only_s42.yaml
│   └── baselines/
│       ├── mult_lite_s42.yaml
│       ├── selfmm_lite_s42.yaml
│       ├── tfn_lite_s42.yaml
│       ├── lmf_lite_s42.yaml
│       ├── misa_lite_s42.yaml
│       ├── mlcl_lite_s42.yaml
│       └── dlf_lite_s42.yaml
└── README.md
```

## 5. Data Paths

| Dataset | Path |
|---------|------|
| MOSI labels | `data/mosi/label.csv` |
| MOSI features | `data/features_strong_sequence_mosi_v3_T40/` |
| MOSEI labels | `data/processed/mosei_full/label.csv` |
| MOSEI features | `data/processed/mosei_full/` |
| MOSEI RoBERTa cache | `data/processed/mosei_full/roberta_cache/` |

## 6. One-Click Matrix Script

```bash
#!/bin/bash
# scripts/p6u_freeze_all.sh
cd E:\00project_code\main_leo\new_code
conda activate mme

# MOSEI Baselines (7 models, ~2 min each)
for model in mult_lite selfmm_lite tfn_lite lmf_lite misa_lite mlcl_lite dlf_lite; do
  python scripts/train_baseline_lite.py \
    --config configs/experiments/p6u_freeze/mosei/baselines/${model}_s42.yaml \
    --device cuda
done

# MOSI Baselines (4 models)
for model in mult_lite selfmm_lite tfn_lite lmf_lite; do
  python scripts/train_baseline_lite.py \
    --config configs/experiments/p6u_freeze/mosi/${model}_s42.yaml \
    --device cuda
done

# MOSEI Main Model
python scripts/train_textft_lora_mainline.py \
  --config configs/experiments/p6u_freeze/mosei/main_text_audio_s42.yaml \
  --device cuda

echo "P6U freeze matrix complete"
```

## 7. Smoke Config

Each model: `--smoke` (1 epoch, 5 batches). Expected: no NaN, no crash, predictions non-constant.

## 8. Real Run Config

Each model: full epochs per config (12 for baselines, 12-20 for main). Multi-seed if time allows.

## 9. Figure/Table Output Plan

| Artifact | Content | Script |
|----------|---------|--------|
| Table 1 | MOSI main + baseline results | manual / aggregate script |
| Table 2 | MOSEI main + baseline results | manual / aggregate script |
| Table 3 | Ablation (if run) | from ablation configs |
| Figure 1 | AWAF weight distribution | `save_plots()` in main script |
| Figure 2 | Training curves | `save_plots()` in main script |
| Figure 3 | Confusion matrix | `save_plots()` in main script |
| Figure 4 | Model architecture diagram | external (draw.io / TikZ) |

## 10. File Cleanup Plan

Objects to clean (in P6U, with manifest):
- `outputs/P6H_repair/` (early MOSI experiments, superseded)
- `outputs/P6I/` (intermediate, superseded by P6K)
- `outputs/P6N/` through `outputs/P6R/` (MOSI baseline intermediate)
- `outputs/P6S_repair/` through `outputs/P6S_repair4/` (MOSEI exploration)
- `outputs/P4*/`, `outputs/P5*/` (early development)
- `outputs/_invalid/` (quarantined collapse outputs)

Keep:
- `outputs/P6K/` (MOSI main model best)
- `outputs/P6S_repair5/` (MOSEI baselines best)
- `outputs/P6T_prefreeze/` (MOSEI main model)
- `outputs/P6U_freeze/` (final freeze runs)
- All source code, configs, reports

## 11. Git Plan

| Step | Action |
|------|--------|
| 1 | Commit all P6T reports + configs |
| 2 | Commit modified source files |
| 3 | Create branch `p6u-model-freeze` |
| 4 | Run freeze scripts on new branch |
| 5 | Commit freeze outputs |
| 6 | Create PR to main |

## 12. Timeline Estimate

| Task | Time |
|------|------|
| Complete MOSEI main model (current) | ~1.5h |
| TAV smoke (optional) | ~10 min |
| P6U freeze re-run (all models) | ~3h |
| Table/figure generation | ~1h |
| Cleanup + git | ~30 min |
| **Total P6U** | **~6h** |

Generated: 2026-06-19
