# P6W-C: MOSI Canonical — Permanently Blocked

**Date**: 2026-06-20  
**Status**: `failed_collapse_under_current_protocol`

## Attempt Summary

| Attempt | Strategy | Epochs | Result |
|---------|----------|--------|--------|
| 1 | End-to-end (P6W-C code) | 11 (early stop) | ACC2=42.38% collapse |
| 2 | Staged (freeze text, train AWAF) | 8 (killed) | val stuck 57.41% collapse |

## Evidence

Both attempts show identical collapse pattern:
- val_ACC2 stuck at 57.41-57.87% = MOSI val pos/non0
- test ACC2 = 42.23-42.38% = MOSI test pos/non0
- Model predicts all-positive regardless of fusion/encoder config
- Loss barely decreases: 1.51 → 1.31 (only 13% reduction over 8 epochs)

## Root Cause

MOSI (1284 training samples) is insufficient for canonical AWAF+sLSTM training:
- 358M RoBERTa parameters create optimization landscape too difficult for 20 optimizer updates/epoch
- Even with frozen text (Attempt 2), AWAF+audio head cannot learn meaningful fusion from 1284 samples
- MOSEI (14700 samples) succeeds because 11.5× more data supports multimodal co-training

## Disposition

- P6K 88.72%: `P6K_conservative_reference` — NOT an AWAF+sLSTM result
- MOSI Canonical: `failed_collapse_under_current_protocol`
- `can_enter_paper: false`
- No Attempt 3 permitted

## Paper Treatment

MOSI serves as a dataset limitation case study:
- P6K 88.72% demonstrates strong text-only performance on small datasets
- Canonical AWAF+sLSTM fails to converge on 1284 samples
- This limitation is documented in Chapter 5 (limitations section)
- MOSEI Canonical ablation provides the formal evidence for AWAF effectiveness

Generated: 2026-06-20
