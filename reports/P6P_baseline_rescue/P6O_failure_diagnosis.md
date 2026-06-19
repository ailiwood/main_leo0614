# P6O Baseline-Lite Failure Diagnosis

## Key Findings

| Finding | Detail |
|---------|--------|
| Majority baseline | test ACC2_Non0 = 57.8% |
| All 4 P6O models BELOW majority | TFN 42%, LMF 48%, MulT 44%, SelfMM 50% |
| Sign flip | ALL 4 models better when predictions are flipped → output sign is wrong |
| Prediction collapse | TFN: 100% preds positive, std=0.09 |
| Text-only probe (random embed) | test=65.2% — already beats all P6O models |
| 200-sample overfit | SelfMM reaches 100% — model CAN learn |

## Root Causes

1. **hidden_dim=32 too small** — insufficient capacity for 1284 samples
2. **Random embedding, no pretrained text** — text probe proves random embed + MLP (65%) > P6O baselines (42-50%)
3. **No classification loss** — only L1 regression, ACC2 needs sign match
4. **Model overfits early (E2-E4)** then never improves — learning rate / capacity mismatch
5. **All models collapse to predicting same sign** — bias term dominates

## Fix Plan

1. Use cached/frozen RoBERTa features for text
2. hidden_dim=128
3. Add BCE classification loss
4. Retrain SelfMM-lite and MulT-lite
