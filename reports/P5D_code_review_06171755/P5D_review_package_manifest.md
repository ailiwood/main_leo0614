# P5D Review Package Manifest

## Package Info

| Item | Value |
|------|-------|
| Zip path | `review_packages/P5D_code_review_06171755.zip` |
| Zip size | 89 KB |
| File count | 44 |
| Prohibited files | None (verified: no pth/pt/ckpt/wav/mp4/npz/npy/pkl) |
| Created | 2026-06-17 17:55 |

## Contents

### code/ (21 files)

```
code/models/deeptext_xlstm_awaf_residual.py
code/models/encoders/slstm.py
code/models/fusion/awaf.py
code/models/pooling/attention_pooling.py
code/models/interaction/cross_modal_transformer.py
code/models/heads.py
code/models/modules/conditional_residual_gate.py
code/engine/losses.py
code/engine/strict_trainer.py
code/utils/metrics.py
code/utils/seed.py
code/scripts/train_deeptext_xlstm_awaf_residual.py
code/scripts/train_deeptext_xlstm_awaf_residual_twostage.py
code/scripts/test_deeptext_xlstm_awaf_residual.py
code/scripts/analyze_residual_effect.py
code/configs/models/deeptext_xlstm_awaf_residual_mosi.yaml
code/configs/models/deeptext_xlstm_awaf_residual_mosei_sdk.yaml
code/configs/data/mosi_strong_sequence.yaml
code/data/strong_sequence_dataset.py
```

### reports/ (14 files)

```
reports/HANDOFF_PHASE_11C.md
reports/HANDOFF_PHASE_12D.md
reports/P5D_seed2024_result.md
reports/P5D_2seed_summary.csv
reports/P5D_residual_effect_analysis.md
reports/P5D_checkpoint_space_audit.md
reports/FINAL_MODEL_SPEC_DRAFT.md
reports/主模型大修决策06171310.md
reports/DECISIONS.md
reports/memory.md
reports/CODE_REVIEW_ENTRYPOINT.md
```

## Suggested Upload to Web AI

1. `review_packages/P5D_code_review_06171755.zip` (89KB — all source + reports)
2. `HANDOFF_CODE_REVIEW_06171755.md` (this handoff)
3. `reports/P5D_code_review_06171755/P5D_results_and_open_questions.md`
4. `reports/P5D_code_review_06171755/P5D_key_code_excerpts.md`
5. `reports/P5D_code_review_06171755/P5D_code_review_summary.md`
