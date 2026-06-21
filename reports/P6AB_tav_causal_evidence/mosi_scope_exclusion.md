# MOSI Scope Exclusion — P6AB

**Date**: 2026-06-21  
**MOSI_STATUS**: `excluded_from_new_training`  
**REASON**: Repeated canonical instability and insufficient marginal evidence value  
**PAPER_ROLE**: `historical_text_audio_reference_only`  
**GPU_BUDGET**: Zero

---

## Decision

MOSI is excluded from all P6AB training, ablation, feature extraction, and analysis.

### What was NOT done

- No MOSI TAV feature contract audit
- No MOSI vision data extraction
- No MOSI TAV dataset build
- No MOSI TAV main control training
- No MOSI TAV ablation training
- No MOSEI→MOSI checkpoint transfer
- No MOSI-related GPU cycles consumed

### What MOSI remains

MOSI retains exactly one historical reference:

| Model | Phase | ACC2_Non0 | F1_Non0 | Classification |
|-------|-------|-----------|---------|----------------|
| P6K MOSI T+A conservative | P6K | 88.72% | 86.64% | `historical_text_audio_reference` |

### Paper language (mandatory)

If MOSI is mentioned in the paper:

> CMU-MOSI is retained as a historical text-audio conservative reference.  
> As the current Canonical three-modal model did not establish a stable,  
> reproducible training pipeline under MOSI's small-sample setting,  
> MOSI is not used as formal experimental evidence for the three-modal main model.

### Prohibitions

1. Do NOT write MOSI 88.72% as T+A+V or AWAF+sLSTM result
2. Do NOT put MOSI and MOSEI TAV in the same "main model improvement" column
3. Do NOT restart MOSI training to "complete dual-dataset evidence"
4. Do NOT put MOSI collapse results in main tables
5. Do NOT allocate GPU to MOSI
6. Do NOT block P6AB on MOSI feature contract

### MOSEI as sole evidence dataset

MOSEI is the only dataset providing formal T+A+V evidence:
- Complete-case TAV cohort (`mosei_official_tav_intersection_v1`)
- 7-variant causal ablation matrix
- Same-cohort, same-budget controlled comparison
- AWAF weight, counterfactual, and statistical analysis
