# P6C Stage Decision

## BREAKTHROUGH: RoBERTa-large text-only = 85.37%

| Model | ACC2 | MAE | Corr |
|-------|------|-----|------|
| Frozen DeBERTa text-only (P5B) | 80.2% | 0.885 | 0.733 |
| **RoBERTa-large fine-tuned text-only** | **85.37%** | **0.6460** | **0.8256** |
| Δ | **+5.17%** | **-0.239** | **+0.093** |

## Path to 87+

RoBERTa text-only (85.37%) + xLSTM-AWAF residual (+1-2%) = **86-87%** target range.

## Decision

1. ✅ RoBERTa-large fine-tuning is the real solution
2. ✅ Next: Build TextFT-xLSTM-AWAF Residual model
3. ✅ Target: Text-only 85.37% + residual → 87%

## Failed Routes (Confirmed)
- wav2vec2-large: 80.18% ❌
- CLIP ViT-L/14: 79.12% ❌
- Frozen DeBERTa: 80.2% ❌
