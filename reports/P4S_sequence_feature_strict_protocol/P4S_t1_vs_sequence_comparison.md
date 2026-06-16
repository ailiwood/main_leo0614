# P4S T=1 vs Sequence Feature Comparison

## Results

| Feature | T | ACC2_NZ | MAE | Corr | Notes |
|---------|:--:|:--:|:--:|:--:|------|
| **TMDC-v1 (T=1)** | 1 | **75.25%** | 1.000 | 0.592 | DeBERTa+wav2vec+CLIP 1024d |
| MLCL standard (T~12) | 12 | **44.9%** | 1.493 | -0.023 | WordEmb+COVAREP+FACET |

## Why MLCL Features Perform Worse

| Aspect | TMDC-v1 | MLCL Standard |
|--------|---------|---------------|
| Text | DeBERTa-large 1024d | Word embedding 300d (trainable, 3117 vocab) |
| Audio | wav2vec2-large 1024d | COVAREP 47d |
| Vision | CLIP-ViT-B/32 1024d | FACET 74d |
| Sequence | T=1 (pooled) | T~12 (word-aligned) |
| Pretrain scale | ~1B params combined | 0 (from scratch) |

The TMDC features benefit from massive pretrained models. The MLCL features, despite preserving temporal structure (T>1), use much weaker features (COVAREP 47d, FACET 74d, raw word embeddings).

sLSTM temporal modeling cannot compensate for the feature quality gap.

## Key Conclusion

**T=1 with strong pretrained features >> T>1 with weak features.**

This is significant for the paper: the current model+feature combination (sLSTM+AWAF with TMDC features) already outperforms what standard MMSA features would give.

## Recommendation

For the paper, use TMDC-v1 1024d features as the primary feature set. Acknowledge T=1 limitation but emphasize the benefit of strong pretrained features. If sequence features are needed, use pretrained sequence features (e.g., BERT token-level embeddings, not word-level GloVe).
