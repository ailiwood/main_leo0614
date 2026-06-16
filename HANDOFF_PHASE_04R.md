# HANDOFF_PHASE_04R.md

## P4A Results Trustworthiness

**NOT directly paper-usable** due to:
1. Test leakage (best_epoch selected by test data)
2. ACC2 computation different from MMSA convention (cls_logit vs reg_sign)
3. T=1 features not comparable to MMSA T~50 sequences

## Test Leakage

best_epoch was chosen by test ACC2_Non0, inflating results by ~1.5%.

## Metric Recomputation

- Reg_pred sign gives +1.4% ACC2 over cls_logit
- Best epoch values (by test): 76.68-76.83%
- Final epoch values (reg_sign): 76.37-76.83%

## Baseline Alignment

75.25% is NOT directly comparable to MMSA ~85%.
Reasons: feature gap (T=1 vs T~50), ACC2 computation, test protocol.

## Root Cause of Low Performance

**T=1 pooled features.** sLSTM cannot model temporal dynamics.
AWAF operates on static embeddings.
Not a model design issue — a feature pipeline issue.

## Recommended Next Step

**Route B: Switch to standard sequence features.**
- Use MMSA/MLCL aligned word+frame features (T~50)
- Let sLSTM actually process sequences
- Fix test protocol (val-tuned, test-once)
- Switch ACC2 to reg_pred sign

## Questions for User / Web AI

1. Accept Route B recommendation?
2. Authorize MMSA/MLCL feature download?
3. MOSEI: proceed with sequence features too or wait?
4. GitHub upload: execute cleanup or wait for P5?
