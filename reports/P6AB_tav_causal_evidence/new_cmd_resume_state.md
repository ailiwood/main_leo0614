# P6AB: New CMD Resume State Report

**Timestamp**: 2026-06-21 03:07 UTC  
**Session**: New Claude Code window, fresh CMD session  
**Working Directory**: `E:\00project_code\main_leo\new_code`

---

## 1. Current Environment State

| Item | Value |
|------|-------|
| Python | 3.11.7 (system Python, NOT conda `mme` env) |
| Python path | `C:\Users\Windows11\AppData\Local\Programs\Python\Python311\python` |
| Git branch | `p6aa-mosei-vision-recovery-tav` |
| Git commit | `604b773` — "P6AA: fix sLSTM mask restore on first timestep (h_prev_cp=None)" |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16GB VRAM |
| GPU memory used | ~1960 MiB / 16303 MiB (desktop apps only, no training) |
| Running Python training processes | **0** (verified via `ps aux`) |
| Conda env `mme` | NOT activated — need to activate before training |

## 2. Git Status — Working Directory NOT Clean

### Modified tracked files (8):
```
M configs/baselines/mosei/dlf_lite_s42.yaml
M configs/baselines/mosei/lmf_lite_s42.yaml
M configs/baselines/mosei/misa_lite_s42.yaml
M configs/baselines/mosei/mlcl_lite_s42.yaml
M configs/baselines/mosei/mult_lite_s42.yaml
M configs/baselines/mosei/selfmm_lite_s42.yaml
M configs/baselines/mosei/tfn_lite_s42.yaml
M models/textft_lora_xlstm_awaf_residual.py
```

### Untracked files (4):
```
?? P6Z_双数据集快速因果消融实验设计.md
?? configs/ablation_fast_causal/
?? docs/P6Z_双数据集快速因果消融实验设计.md
?? scripts/build_mosei_tav_openface2_v1.py
```

### Nature of modifications:
- **7 baseline configs**: Changed from `text_audio` + `mosei_full` → `text_audio_vision` + `mosei_tav_openface2_v1` + `cohort_id: mosei_official_tav_intersection_v1` + `vision_input_dim: 713`
- **model file**: Added `canonical_text_audio_vision_awaf_slstm` mode support; fixed GRU/LSTM bidirectional→unidirectional; added vision branch routing for TAV canonical mode
- **Untracked**: P6Z experiment design docs, P6Z ablation configs (mosei + mosi), build script for TAV dataset

### Verdict:
All modifications are **P6AA continuation/in-progress work** (not committed to P6AA branch yet). The baseline configs were being upgraded for TAV but not yet trained. The model changes are essential for P6AB TAV work. Do NOT discard or reset.

## 3. P6AA TAV Artifact Inventory

### 3.1 P6AA Main Model (TAV Canonical)
| Field | Value |
|-------|-------|
| Config | `configs/experiments/p6aa_tav/mosei/control_awaf_slstm_s42.yaml` |
| Phase | P6AA_TAV |
| Mode | `canonical_text_audio_vision_awaf_slstm` |
| Data version | `mosei_tav_openface2_v1` |
| Cohort ID | `mosei_official_tav_intersection_v1` |
| Epochs | 12 |
| Best epoch | 11 (val ACC2 = 87.72%) |
| **Final ACC2_Non0** | **87.978%** |
| F1_Non0 | 90.49% |
| MAE | 0.533 |
| Corr | 0.785 |
| ACC7 | 53.92% |
| Trainable params | 4.84M |
| AWAF w_t avg | 0.481 |
| AWAF w_a avg | 0.264 |
| AWAF w_v avg | 0.254 |
| Output path | `outputs/P6AA_tav/mosei/main/control_awaf_slstm_s42_s42_20260620_223047/` |
| Best model | `best_model.pth` (1.4 GB) |
| Predictions CSV | `predictions_test.csv` (4221 samples) |
| AWAF weights CSV | `awaf_weights_test.csv` |

### 3.2 P6AA Run History
7 timestamped subdirectories in `outputs/P6AA_tav/mosei/main/`. Only 2 produced result.json:
- `222728`: 1 epoch only, ACC2 = 61.96% (warmup/aborted)
- `223047`: 12 epochs, ACC2 = 87.98% (complete run)

Earlier runs (222312–222654) have no result.json — likely crashed or aborted early.

### 3.3 P6AA Baseline Models
**Status: NOT YET TRAINED.** Configs exist but no output directories found. The 7 baseline configs have been modified for TAV but no training runs were executed.

### 3.4 P6AA Reports
**Status: EMPTY.** `reports/P6AA_mosei_vision_recovery/` directory exists but contains 0 files. No audit reports were generated for P6AA.

## 4. Historical Reference Results

| Model | Phase | Mode | Epochs | Cohort | ACC2_Non0 | Notes |
|-------|-------|------|--------|--------|-----------|-------|
| P6S text-only | P6S_repair | text_only | 20 | mosei_full | 88.22% | Different cohort, 20 epochs |
| P6T T+A mainline | P6T_prefreeze | text_audio_residual | 12 | mosei_full (?) | 87.86% | gate_mean=1.0, delta=0 |
| P6Z F0 T+A sLSTM | P6Z_fast_causal | text_audio_awaf_slstm | 4 | unknown | 63.15% | 4-epoch budget, gate=0 |
| **P6AA TAV main** | **P6AA_TAV** | **tav_awaf_slstm** | **12** | **tav_intersection** | **87.98%** | **Current reference** |

**Critical comparability limitation**: These results use DIFFERENT cohorts, data versions, epochs, and architectures. They CANNOT be directly compared to produce "vision gain" claims.

## 5. Key Technical Findings from Initialization

### 5.1 Model architecture is correct
- The canonical TAV model properly encodes all three modalities through sLSTM
- AWAF produces sample-specific 3-modal weights (4215 unique w_t values out of 4221 samples)
- AWAF weight distribution: text dominates (~48%), audio (~26%), vision (~25%)
- The model produces 4221 unique predictions → NOT a collapsed/constant predictor

### 5.2 gate=0.0 and delta=0.0 are NOT bugs
- Canonical TAV mode does NOT use gate/delta mechanism (it's for residual architecture modes)
- `gate=0.0` and `delta=0.0` in predictions CSV are default fill values
- `text_base_ACC2 == final_ACC2` because canonical mode returns same value for both fields

### 5.3 No independent text baseline on TAV cohort
- We cannot measure vision/audio contribution without a separate text-only model on the same cohort
- The current `residual_gain=0.0` is an artifact, not evidence of zero contribution

### 5.4 Baseline configs modified but not trained
- 7 baseline configs have been upgraded to TAV (text_audio_vision + vision_dim=713)
- These need to be trained to serve as TAV baselines

## 6. P6AB Action Items Determined

### Branch strategy:
- Current branch `p6aa-mosei-vision-recovery-tav` has uncommitted P6AA changes
- Create new branch `p6ab-tav-causal-evidence-mosi-contract` from current HEAD
- Preserve all uncommitted changes

### Priority actions:
1. Create P6AB branch from current state
2. Write P6AA result integrity audit (artifact inventory complete above)
3. Build P6AB ablation configs (F0-F5, E1 for MOSEI TAV)
4. Implement switch integrity audit
5. Run 4-epoch ablation matrix
6. Audit MOSI TAV features
7. Generate final evidence report

## 7. Directory Checklist

| Path | Status |
|------|--------|
| `data/processed/mosei_tav_openface2_v1/` | ✅ Exists (train/test/valid with .npz files) |
| `configs/experiments/p6aa_tav/mosei/` | ✅ Exists (1 config: control_awaf_slstm_s42.yaml) |
| `outputs/P6AA_tav/mosei/main/` | ✅ Exists (7 run dirs, 1 complete) |
| `reports/P6AA_mosei_vision_recovery/` | ⚠️ Exists but EMPTY |
| `reports/P6Z_fast_causal_ablation/` | ⚠️ Exists but EMPTY |
| `models/textft_lora_xlstm_awaf_residual.py` | ✅ Modified for TAV support |
| `utils/metrics.py` | ✅ Exists |
| `data/textft_multimodal_dataset.py` | ✅ Exists |
| `scripts/train_textft_lora_mainline.py` | ✅ Exists |
| `reports/P6AB_tav_causal_evidence/` | ✅ Created (this session) |
| `configs/experiments/p6ab_tav_ablation/mosei/` | ✅ Created (this session) |
| `outputs/P6AB_tav_ablation/mosei/` | ✅ Created (this session) |
| `reports/P6AB_tav_causal_evidence/mosei/` | ✅ Created (this session) |
| `reports/P6AB_tav_causal_evidence/mosi/` | ✅ Created (this session) |

## 8. Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| No text-only baseline on TAV cohort | HIGH | Must train F1 (no_vision) on same cohort |
| TAV main uses different data than P6S/P6T | HIGH | Cannot compare directly; need same-cohort baselines |
| Conda env not yet activated | MEDIUM | Activate `mme` before training |
| Python 3.11 vs project Python 3.10/3.9 | MEDIUM | .pyc cache shows 3.10 and 3.9; need conda env |
| Uncommitted baseline config changes | LOW | Preserved; part of P6AB work |
| No previous P6AA audit reports | MEDIUM | P6AB-0 audit will fill this gap |
| AWAF canonical mode doesn't support residual measurement | HIGH | Need to add text-baseline measurement for ablation |
