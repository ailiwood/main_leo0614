# P5C MOSI Seed42 60-Epoch Result Report

> Date: 2026-06-17
> Model: DeepText-xLSTM-AWAF Residual (ablation=none)
> Seed: 42, Batch: 32, LR: 1e-4
> Features: v3_T40 (text=1024, audio=768, vision=768)
> Config: configs/models/deeptext_xlstm_awaf_residual_mosi.yaml

## Final Test Results (best epoch=38 by val MAE)

### Primary Metrics (reg_sign)

| Metric | Value | vs DeepMLP text-only | vs P4W AWAF-Seq |
|--------|-------|---------------------|-----------------|
| **ACC2_Non0** | **81.10%** | **+0.9%** ✅ | **+2.3%** ✅ |
| F1_Non0 | 77.70% | — | — |
| ACC2_Has0 | 79.45% | — | — |
| MAE | 0.8155 | -0.0695 ✅ | -0.1785 ✅ |
| Corr | 0.7487 | +0.0157 ✅ | +0.1037 ✅ |
| ACC7 | 42.13% | — | — |

### Classification Metrics

| Metric | Value |
|--------|-------|
| ACC2_Non0 (cls) | 80.95% |
| F1_Non0 (cls) | 77.95% |

## Training Dynamics

| Phase | Epochs | Val ACC2 Range | Notes |
|-------|--------|---------------|-------|
| Rapid learning | 1-10 | 57%→80% | Fast convergence |
| Plateau push | 11-30 | 80%→82% | Slow improvement |
| Best region | 31-45 | 82%→84% | Best val at epoch 38 |
| Saturation | 46-60 | 83%→84% | LR reduced to 1.25e-5 |

**Best val ACC2_NZ = 84.26%** at epoch 51 (val-test gap ~3.2%)

## AWAF Weights

| Modality | Mean | Std | Interpretation |
|----------|------|-----|----------------|
| Text (w_t) | 0.1753 | 0.0778 | Text residual contributes ~18% |
| Audio (w_a) | 0.4382 | 0.2781 | Audio is primary residual source |
| Vision (w_v) | 0.3864 | 0.3202 | Vision secondary, high variance |

- sum(w)=1 verified (max dev = 1.19e-7)
- AWAF weights are NOT collapsed — meaningful cross-modal contribution
- Audio dominates residual correction (43.8%), vision close second (38.6%)

## Key Findings

### 1. Residual Architecture VALIDATED ✅
- DeepText-xLSTM-AWAF Residual (81.10%) > DeepMLP text-only (80.2%)
- Multimodal residual fusion IS HELPING, not hurting
- Compared to P4W where multimodal hurt performance (78.8% < 80.2%)

### 2. Regression Quality IMPROVED
- MAE improved from 0.885 (text-only) to 0.8155 (-7.8%)
- Corr improved from 0.733 (text-only) to 0.7487 (+2.1%)
- Residual branch adds meaningful regression fine-tuning

### 3. AWAF Learns Meaningful Weights
- Audio (0.44) and Vision (0.39) share residual responsibility
- High variance suggests per-sample adaptation is working
- Text residual weight is lower (0.18) — text already provides base prediction

### 4. Val-Test Gap
- Best val ACC2 = 84.26%, Test ACC2 = 81.10% (gap ~3.2%)
- Typical for MOSI (small dataset, high variance)
- Could potentially be reduced with stronger regularization

## Judgment

| Criterion | Value | Met? |
|-----------|-------|------|
| ACC2 > 83% | 81.10% | ❌ No |
| ACC2 > 80.2% (DeepMLP) | 81.10% | ✅ Yes (+0.9%) |
| ACC2 > 78.8% (P4W) | 81.10% | ✅ Yes (+2.3%) |
| Between 80.2 and 82 | 81.10% | ✅ Yes |

**Verdict**: ✅ Residual fusion is validated and beneficial. Performance exceeds text-only baseline.
The gain is meaningful but modest (+0.9%). Recommend seed=2024 to confirm stability.
If multi-seed average > 81%, the architecture is confirmed for the paper.

## Outputs
- outputs/P5C/MOSI/deeptext_xlstm_awaf_residual/20260617_170908_s42/
  - best_model.pth, last_model.pth
  - predictions_test.csv, awaf_weights_test.csv
  - test_metrics_final.json, val_metrics_epoch.csv
