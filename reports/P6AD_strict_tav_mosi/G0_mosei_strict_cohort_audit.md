# P6AD-G0: MOSEI Strict Complete-Case Cohort Audit

**Date**: 2026-06-21

## 1. Cohort Summary

| Metric | Value |
|--------|-------|
| Total feature files | 20680 |
| Total label CSV rows | 20680 |
| Duplicate sample IDs | 0 |
| **Strict complete-case valid** | **18571** |
| Excluded | 2109 |
| Target (19,354) match | NO (diff=-783) |

## 2. Per-Split Breakdown

| Split | Valid | Label Mean | Label Std | Pos Ratio | Audio Frames | Vision Frames |
|-------|-------|-----------|-----------|-----------|-------------|--------------|
| train | 13239 | 0.106 | 1.140 | 0.696 | 100.0 | 44.4 |
| valid | 1561 | 0.172 | 1.072 | 0.719 | 100.0 | 44.5 |
| test | 3771 | 0.120 | 1.132 | 0.700 | 100.0 | 44.0 |

## 3. Exclusion Reasons

| Reason | Count |
|--------|-------|
| vision_mask_zero | 1326 |
| audio_nonfinite_2 | 735 |
| audio_nonfinite_1 | 26 |
| audio_nonfinite_4 | 18 |
| audio_nonfinite_6 | 2 |
| audio_nonfinite_3 | 1 |
| audio_nonfinite_5 | 1 |

## 4. P6AA/P6AB Result Classification

**P6AA/P6AB used 20680 samples, but only 18571 pass strict complete-case criteria.**

**Classification**: `aligned_or_mixed_tav_reference`

P6AA/P6AB results CANNOT be claimed as strict complete-case MOSEI TAV results.
They remain valid as aligned/mixed TAV reference points.

**Subsequent MOSEI final tables, baselines, and ablations MUST use**:
`mosei_tav_complete_case_v2` ({total_valid} strict complete-case samples)

## 5. Test Set Vision Quality

| Metric | Value |
|--------|-------|
| Test samples with all-zero vision | 0 |
| Test total valid | 3771 |

## 6. Required Actions

- [ ] Investigate why target is 19,354 but actual is 18571
- [ ] Build `mosei_tav_complete_case_v2` with strict filtered samples
- [ ] Re-run key ablations (F0, F1, F4, F5, E1) on strict cohort
- [ ] Update all paper claims to reference strict cohort
