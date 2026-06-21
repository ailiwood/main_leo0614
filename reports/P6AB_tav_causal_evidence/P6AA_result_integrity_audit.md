# P6AA TAV Result Integrity Audit

**Date**: 2026-06-21  
**Auditor**: P6AB initialization phase  
**Branch**: `p6ab-tav-causal-evidence-mosi-contract`  
**Base commit**: `604b773`

---

## 1. Executive Summary

| Verdict | Status |
|---------|--------|
| P6AA TAV main artifacts | `formal_complete` |
| P6AA TAV baseline artifacts | `missing_artifact` — configs modified but not trained |
| P6AA audit reports | `missing_artifact` — reports dir exists but empty |
| TAV vs old T+A comparability | `not_directly_comparable` — different modalities, cohorts partially different |
| Canonical mode text_base measurement | `invalid` — canonical mode returns same value for text_base and final |

### Key Finding

**P6AA has proven the TAV data pipeline and vision computation path are functional.** The model produces 4221 unique predictions with sample-specific AWAF weights (w_t≈48%, w_a≈26%, w_v≈25%). However, **the current canonical architecture cannot independently measure vision contribution** because `reg_text_base ≡ reg_final` in canonical mode.

---

## 2. Artifact Completeness Audit

### 2.1 P6AA TAV Main Model

| Artifact | Path | Status | Notes |
|----------|------|--------|-------|
| Config YAML | `configs/experiments/p6aa_tav/mosei/control_awaf_slstm_s42.yaml` | ✅ | 12 epochs, s42, TAV canonical |
| Best model | `control_awaf_slstm_s42_s42_20260620_223047/best_model.pth` | ✅ | 1.4 GB, epoch 11 |
| Last model | Not saved separately | ⚠️ | Only best_model saved |
| Result JSON | `result.json` | ✅ | ACC2=87.98%, F1=90.49%, MAE=0.533, Corr=0.785 |
| Metrics CSV | `metrics_epoch.csv` | ✅ | All 12 epochs tracked |
| Predictions CSV | `predictions_test.csv` | ✅ | 4221 samples, 12 columns |
| AWAF weights CSV | `awaf_weights_test.csv` | ✅ | Per-sample 3-modal weights |
| Group error | `group_error_analysis.csv` | ✅ | 5 sentiment groups |
| Training curves | `mosi_training_curves.png` | ✅ | Loss + metrics plot |
| AWAF weight plot | `mosi_awaf_weight_distribution.png` | ✅ | Weight distribution histogram |
| Confusion matrix | `mosi_confusion_matrix.png` | ✅ | Confusion matrix plot |
| Delta distribution | `mosi_delta_distribution.png` | ⚠️ | All zeros (no delta in canonical mode) |
| Gate distribution | `mosi_gate_distribution.png` | ⚠️ | All zeros (no gate in canonical mode) |
| Scatter plot | `mosi_text_vs_final_scatter.png` | ✅ | text_base vs final (identical line) |
| Command log | `command.txt` | ❌ | Not saved |
| Model signature | `model_signature.json` | ❌ | Not saved |
| Gradient audit | `gradient_audit.json` | ❌ | Not saved |
| Config SHA256 | Not computed | ❌ | Missing |

### 2.2 P6AA TAV Baselines (7 models)

| Model | Config | Output | Status |
|-------|--------|--------|--------|
| DLF-lite | `configs/baselines/mosei/dlf_lite_s42.yaml` (M) | None | `missing_artifact` |
| LMF-lite | `configs/baselines/mosei/lmf_lite_s42.yaml` (M) | None | `missing_artifact` |
| MISA-lite | `configs/baselines/mosei/misa_lite_s42.yaml` (M) | None | `missing_artifact` |
| MLCL-lite | `configs/baselines/mosei/mlcl_lite_s42.yaml` (M) | None | `missing_artifact` |
| MULT-lite | `configs/baselines/mosei/mult_lite_s42.yaml` (M) | None | `missing_artifact` |
| SelfMM-lite | `configs/baselines/mosei/selfmm_lite_s42.yaml` (M) | None | `missing_artifact` |
| TFN-lite | `configs/baselines/mosei/tfn_lite_s42.yaml` (M) | None | `missing_artifact` |

**(M)** = Modified in working tree: upgraded from `text_audio` + `mosei_full` to `text_audio_vision` + `mosei_tav_openface2_v1`.

### 2.3 P6AA Reports

| Report | Path | Status |
|--------|------|--------|
| Final report | `reports/P6AA_mosei_vision_recovery/P6AA_final_report.md` | `missing_artifact` |
| Source audit | `reports/P6AA_mosei_vision_recovery/A0_mosei_openface2_source_audit.md` | `missing_artifact` |
| Feature build report | `reports/P6AA_mosei_vision_recovery/A1_tav_feature_build_report.md` | `missing_artifact` |
| Quality gate | `reports/P6AA_mosei_vision_recovery/A2_tav_dataset_quality_gate.md` | `missing_artifact` |
| Static analysis | `reports/P6AA_mosei_vision_recovery/G1_TAV_static.md` | `missing_artifact` |
| Dynamic analysis | `reports/P6AA_mosei_vision_recovery/G2_TAV_dynamic.md` | `missing_artifact` |
| Overfit analysis | `reports/P6AA_mosei_vision_recovery/G3_TAV_overfit200.md` | `missing_artifact` |
| Health check | `reports/P6AA_mosei_vision_recovery/TAV_main_health_check.md` | `missing_artifact` |
| Baseline audit | `reports/P6AA_mosei_vision_recovery/tav_baseline_consumption_audit.md` | `missing_artifact` |
| Experiment registry | `reports/experiment_registry.csv` | `missing_artifact` |

---

## 3. Data Pipeline Integrity

### 3.1 TAV Feature Verification

| Check | Result |
|-------|--------|
| Feature file format | `.npz` with keys `[audio_seq, audio_mask, vision_seq, vision_mask]` |
| Audio seq shape | (100, 74) — COVAREP features, 100 time steps |
| Vision seq shape | (50, 713) — OpenFace2 features, 50 time steps |
| Audio mask validity | All 100 steps valid per sample |
| Vision mask validity | 50 steps per sample |
| Vision NaN rate | 0% (tested 100 samples) |
| Vision Inf rate | 0% (tested 100 samples) |
| Audio NaN rate | 0% (tested 100 samples) |
| Audio Inf rate | 5% (cleaned at load time by `_clean_features()`) |
| Vision all-zero rate (test) | 6.8% (286/4221 samples) |
| Audio all-zero rate (test) | 0% |
| Sample count (train/valid/test) | 14700 / 1759 / 4221 |
| Sample ID overlap with full data | 100% (identical sample sets) |

### 3.2 Data Version Comparison

| Property | `mosei_tav_openface2_v1` | `mosei_full` | `mosei_text_audio` |
|----------|--------------------------|--------------|---------------------|
| Vision dim | 713 (OpenFace2) | 768 (original) | N/A |
| Vision seq len | 50 | 40 | N/A |
| Audio dim | 74 | 74 | 74 |
| Sample count | 20680 (full) | 20680 | 0 npz files! |
| Feature files | Yes (all splits) | Yes (all splits) | No (only label.csv) |

### 3.3 Model-Data Interface

| Check | Result |
|-------|--------|
| Config `vision_dim: 713` → model `vision_input_dim` | ✅ Correctly mapped via `train_textft_lora_mainline.py:59` |
| Config `audio_dim: 74` → model projection | ✅ Matches feature dimension |
| Vision projection | `nn.Linear(713, 256)` |
| Audio projection | `nn.Linear(74, 256)` |
| Dataset reads correct feature dir | ✅ `mosei_tav_openface2_v1/{split}/` |
| `formal_mode: true` prevents zero fallback | ✅ |

---

## 4. Model Architecture Audit (TAV Canonical)

### 4.1 Prediction Integrity

| Check | Result |
|-------|--------|
| Unique predictions / total | 4221 / 4221 (100%) |
| Prediction range | [-2.14, +2.29] (MOSEI label range [-3, +3]) |
| Constant predictor? | No — 4221 unique values |
| NaN predictions? | Not detected |
| AWAF weights unique w_t | 4215 / 4221 (99.9%) |
| AWAF w_t range | [0.364, 0.549] |
| AWAF w_a range | [0.210, 0.308] (estimated) |
| AWAF w_v range | [0.217, 0.279] (estimated) |

### 4.2 Architecture Validation

| Check | Result |
|-------|--------|
| Text branch active | ✅ RoBERTa-large + LoRA |
| Audio branch active | ✅ sLSTM temporal encoder |
| Vision branch active | ✅ sLSTM temporal encoder |
| AWAF fusion active | ✅ Sample-specific weights observed |
| Gate mechanism | ❌ Not used in canonical mode |
| Delta/residual | ❌ Not used in canonical mode |
| text_base = final? | ✅ By design (canonical mode) |

### 4.3 Residual Gain Analysis

The `residual_gain: 0.0` and `text_base_ACC2 == final_ACC2` are **by design** in canonical TAV mode:
- `_forward_canonical_tav_awaf()` returns `reg_text_base: reg` (same tensor as final)
- This makes `residual_gain` measurement impossible without architecture change
- **This is NOT a bug** — it's a limitation of the canonical architecture for ablation purposes

---

## 5. Cross-Phase Comparability Analysis

### 5.1 Historical Results Comparison

| Model | Phase | Mode | Epochs | Cohort/Data | ACC2_Non0 |
|-------|-------|------|--------|-------------|-----------|
| Text-only | P6S | text_only | 20 | mosei_full | 88.22% |
| T+A Residual | P6T | text_audio_residual | 12 | mosei_full | 87.86% |
| T+A AWAF sLSTM | P6Z | text_audio_awaf_slstm | 4 | unknown | 63.15% |
| **TAV AWAF sLSTM** | **P6AA** | **tav_awaf_slstm** | **12** | **tav_intersection** | **87.98%** |

### 5.2 Comparability Verdict

| Comparison | Verdict | Reason |
|-----------|---------|--------|
| P6AA TAV vs P6S text-only | `not_directly_comparable` | Different epochs (12 vs 20), different modality, P6S has no vision path at all |
| P6AA TAV vs P6T T+A | `not_directly_comparable` | Different data version (TAV vs full), different architecture (canonical vs residual) |
| P6AA TAV vs P6Z F0 | `not_directly_comparable` | Different epochs (12 vs 4), different modalities (TAV vs TA), different data cohort |

### 5.3 Required for Comparability

To claim "vision contributes X% improvement":
1. Train text-only baseline on **same TAV cohort** (mosei_official_tav_intersection_v1)
2. Train T+A baseline on **same TAV cohort**
3. Use **same 12-epoch training budget**
4. Use **same seed (42)**, batch, optimizer, scheduler
5. Compare on **identical test sample set**

**None of these conditions are currently met.**

---

## 6. Integrity Flags

### 6.1 Critical Flags

| Flag | Severity | Description |
|------|----------|-------------|
| No separate text baseline on TAV cohort | **HIGH** | Cannot measure vision contribution |
| Canonical mode conflates text_base with final | **HIGH** | `residual_gain` is structurally 0 |
| Old T+A baselines use different data | **MEDIUM** | Cannot compare across phases |
| 6.8% test samples have all-zero vision | **MEDIUM** | Vision provides no signal for these |
| No command.txt or model_signature.json | **LOW** | Reduces reproducibility |
| sample_id stored as sequential index | **LOW** | Cannot trace prediction to original video |

### 6.2 Non-Issues (Verified)

| Check | Status | Notes |
|-------|--------|-------|
| gate=0.0 is a bug | ❌ NOT a bug | Canonical mode doesn't use gate mechanism |
| delta=0.0 means zero contribution | ❌ NOT true | Canonical mode has no separate text baseline |
| AWAF weights are constant | ❌ NOT true | 4215 unique w_t values observed |
| Model produces constant output | ❌ NOT true | 4221 unique predictions |

---

## 7. Recommendations for P6AB

### 7.1 Immediate Actions
1. **Build independent text-only baseline** on TAV cohort for proper ablation comparison
2. **Add text_base forward pass** to canonical TAV mode for residual measurement
3. **Audit 6.8% zero-vision samples** — check if they degrade TAV performance
4. **Save model_signature.json** in all future runs

### 7.2 Do NOT
- Claim "vision improves ACC2 by +0.15pp" (P6AA TAV 87.98% vs old T+A 87.83%)
- Use P6S text-only 88.22% as text baseline for TAV cohort comparison
- Present P6Z 4-epoch results alongside P6AA 12-epoch results as fair comparison
- Delete P6AA artifacts

### 7.3 P6AB Ablation Design Implications
- P6AB F1 (no_vision) must serve as proper text+audio baseline on same cohort
- P6AB F2 (no_audio) must serve as proper text+vision baseline
- All ablation variants must use 4-epoch fixed budget for internal comparability
- P6AA 12-epoch main remains the canonical candidate, NOT to be compared numerically with 4-epoch ablation
