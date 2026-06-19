# P6S-Repair-5: MOSEI Collapse Diagnosis + Clean Restart — Final Report

**Phase**: P6S-Repair-5  
**Date**: 2026-06-19  
**Status**: ✅ Fix confirmed, 2 baselines completed, 6 running

---

## 1. Killed Processes
**None.** No python training processes were running when this session started. All 8 P6S-Repair-4 baseline runs had already completed (5 epochs each, all collapsed).

## 2. Isolated Outputs
**12 runs isolated** to `outputs/_invalid/P6S_repair4_collapse_20260619_190122/`:
- 8 baseline results (MulT ×5 runs, TFN, LMF, MISA, SelfMM, MMIM, MLCL, DLF)
- All had identical collapse metrics (ACC2=38.04%, F1=0, MAE=NaN, Corr=0)
- Manifest: `reports/P6S_repair5_collapse_fix/invalid_outputs_manifest.csv`
- Registry status: `invalid_collapse`, credibility=D

## 3. Collapse Root Cause
**COVAREP audio features contain `-inf` values** from `log(0)` operations in feature extraction:
- ~2-5% of audio files have 1-2 `-inf` values
- NaN propagation chain: `-inf → Linear → NaN → ReLU → GRU → Cross-attention → Head`
- All 8 different architectures produce identical NaN output
- `torch.where(NaN >= 0, 1.0, -1.0)` → all -1.0 → ACC2 = neg_non0/non0 = 38.0389%

## 4. MAE=NaN Direct Cause
Model regression output is NaN → L1Loss(NaN, labels) = NaN → train loss NaN → no effective gradient → model stays collapsed → predictions all NaN → MAE calculation on all-NaN inputs returns NaN.

## 5. Why 38.04%?
Test set: 4221 samples (2041 pos, 1253 neg, 927 zero). Non0 = 3294.  
Model predicts all-negative → correctly predicts all 1253 negatives → ACC2_Non0 = 1253/3294 = 38.0389%.  
Similarly, valid set: neg_non0/non0 = 481/1360 = 35.3676% (exact match).

## 6. RoBERTa Cache
**✅ Normal.** All splits: shape correct, non-zero embeddings, sample IDs match label.csv.

## 7. Audio Adapter  
**✅ Fixed.** COVAREP 74d features now cleaned by `_clean_features()` in `collate_textft()` — replaces -inf with 0.0.

## 8. Optimizer/Gradient
**✅ Normal.** Total grad norm = 3.56 (not zero). All parameters trainable. Weights update correctly.

## 9. 200-Sample Overfit
**✅ PASS.** Train loss decreased ~40%, ACC2=63.4% (not 38%), F1=75.4% (not 0), no NaN.

## 10. MulT & SelfMM Fixed Results

| Model | ACC2_Non0 | F1_Non0 | MAE | Corr | ACC7 |
|-------|-----------|---------|-----|------|------|
| **MulT-lite** | **82.39%** | **86.17%** | **0.611** | **0.705** | **48.90%** |
| **SelfMM-lite** | **82.60%** | **86.33%** | **0.623** | **0.682** | **49.16%** |

## 11. Remaining 6 Baselines
Launched and training: TFN-lite, LMF-lite, MISA-lite, MMIM-lite, MLCL-lite, DLF-lite.  
All 6 running concurrently on CUDA (RTX 5070 Ti). Results TBD.

## 12. Main Model Impact
**Minimal.** Main model uses the same `collate_textft()` from `data/textft_multimodal_dataset.py`, which is now fixed. The `_clean_features()` function runs automatically on all data paths. If the main model was previously trained on the same data, re-training is recommended.

## 13. Results Usable for Thesis

### Candidate (Confidence A)
| Model | ACC2_Non0 | Source |
|-------|-----------|--------|
| MulT-lite | 82.39% | P6S-Repair-5, Unified metrics |
| SelfMM-lite | 82.60% | P6S-Repair-5, Unified metrics |
| TFN/LMF/MISA/MMIM/MLCL/DLF | TBD | Training in progress |

### Invalidated (Confidence D)
| Model | ACC2_Non0 | Reason |
|-------|-----------|--------|
| ALL P6S-Repair-4 results | 38.04% | Systematic collapse, quarantined |

## 14. Invalidated Results
All P6S-Repair-4 baseline outputs moved to `outputs/_invalid/`. Registry updated to `invalid_collapse`. Not usable for thesis.

## 15. Recovery Commands

### To verify remaining 6 baselines:
```bash
cd E:\00project_code\main_leo\new_code
conda activate mme
# Check outputs:
ls outputs/P6S_repair5/mosei/baselines/
# Check individual results:
cat outputs/P6S_repair5/mosei/baselines/tfn_lite_s42_*/result.json
```

### To launch baseline results aggregation:
```bash
python scripts/aggregate_results.py --phase P6S_repair5 --dataset mosei --output reports/P6S_repair5_baselines_mosei.csv
```

### To run main model with fix:
```bash
python scripts/train_textft_lora_mainline.py --config configs/experiments/p6s_repair5_mosei_mainline_fixed/mosei_main_text_audio_cached_s42.yaml --device cuda
```

### To launch remaining 6 baselines (if not already running):
```bash
for model in tfn_lite lmf_lite misa_lite mmim_lite mlcl_lite dlf_lite; do
  python scripts/train_baseline_lite.py --config "configs/experiments/p6s_repair5_mosei_baselines_fixed/mosei_${model}_cached_s42.yaml" --device cuda &
done
```

---

## Files Modified/Created

### Modified
- `data/textft_multimodal_dataset.py` — Added `_clean_features()` to collate function (fix)
- `utils/metrics.py` — Added NaN fail-fast to `compute_mae()` and `compute_all_metrics()`
- `scripts/train_baseline_lite.py` — Added anti-collapse guard, train_subset support, fixed f-string

### Created
- `tests/test_metrics_mosei_non0.py` — 8 unit tests (all pass)
- `configs/experiments/p6s_repair5_debug/` — Overfit config
- `configs/experiments/p6s_repair5_mosei_baselines_fixed/` — 8 fixed baseline configs
- `configs/experiments/p6s_repair5_mosei_mainline_fixed/` — Main model config dir
- `reports/P6S_repair5_collapse_fix/` — 5 reports
- `outputs/P6S_repair5/mosei/baselines/` — Fixed training outputs

### Isolated
- `outputs/_invalid/P6S_repair4_collapse_20260619_190122/` — 12 collapse runs

---

## Key Lessons
1. **Always check data for inf/NaN before training.** COVAREP's -inf is a known issue.
2. **Metrics should fail-fast on NaN.** Silent corruption produces plausible-looking numbers (38.04% looks like a real result).
3. **Identical metrics across models = systematic bug.** Different architectures cannot produce exactly identical predictions.
4. **CUDA compatibility matters.** RTX 5070 Ti (sm_120) with PyTorch 2.3.0+cu118 works but shows warning. Consider upgrading PyTorch.

Generated: 2026-06-19 19:30 UTC+8
