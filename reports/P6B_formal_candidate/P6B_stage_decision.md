# P6B Stage Decision

## Vision-L/14 V2 Result

| Model | ACC2 | MAE | Corr | vs P5E V2 |
|-------|------|-----|------|-----------|
| P5E V2 baseline | 82.47% | 0.8111 | 0.7385 | — |
| P5G Audio large | 80.18% | 0.8197 | 0.7398 | -2.29% |
| **P6B Vision-L/14** | **79.12%** | **0.8600** | **0.7237** | **-3.35%** |

## Strong Feature Upgrade Summary

Both large-model feature upgrades FAIL on MOSI:
- wav2vec2-large (1024d): -2.29%
- CLIP ViT-L/14 (1024d): -3.35%

**Root cause: Small dataset (1284 training samples) cannot support high-dimensional features. The model overfits.**

Best val ACC2 was 84.26% (similar to baseline) but test performance collapsed, confirming overfitting.

## AWAF evidence

Model actively suppresses strong features:
- Audio large: w_a dropped 0.15→0.03
- Vision large: w_a dropped to 0.02, w_v at 0.26

## Decision

1. ❌ MOSI 83/85/87 NOT reached
2. ✅ Current features (768d base models) are near-optimal for MOSI
3. **Accept P5E V2 (82.47%/82.17%) as best achievable with current architecture**
4. Enter P6C: Formal multi-seed + ablation + baseline + paper preparation
