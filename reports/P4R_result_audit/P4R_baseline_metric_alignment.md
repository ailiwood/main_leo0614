# P4R Baseline Metric Alignment

## Can 75.25% be directly compared to MMSA ~85%?

**NO. Three major incompatibilities:**

### 1. Feature Gap
- Our: T=1 pooled vectors (DeBERTa+wav2vec+CLIP)
- MMSA: T~50 aligned sequences (GloVe/BERT + COVAREP + FACET)
- Text encoder: DeBERTa-large vs GloVe/BERT-base → different semantic quality
- Our features are richer per-timestep but lose all temporal structure

### 2. ACC2 Computation
- Our: Separate classification head (BCE logits)
- MMSA: Regression prediction sign (reg_pred >= 0)
- Switching to reg_sign improves our ACC2 by ~1.4%

### 3. Test Protocol
- Our P4A: best_epoch selected by TEST data (leakage)
- MMSA: val-tuned, test-once protocol
- Fixing protocol reduces our numbers

### 4. Other Factors
- MMSA models (Self-MM, TETFN, CENET) use different architectures
- Our sLSTM+AWAF is unfaired by T=1 limitation
- MOSI is small (1284 train); pretrained weights matter greatly

## Conclusion

**75.25% is NOT directly comparable.** With sequence features + val protocol + reg_sign ACC2, our expected range would be 78-85% depending on feature quality.
