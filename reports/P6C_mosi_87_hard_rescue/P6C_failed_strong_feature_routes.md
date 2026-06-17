# P6C Failed Strong Feature Routes

## wav2vec2-large (1024d audio)
- ACC2: 80.18% (-2.29% vs baseline)
- AWAF w_a: 0.03 (model actively suppressed)
- Verdict: ❌ Closed — 1024d features overfit on small MOSI dataset

## CLIP ViT-L/14 (1024d vision)
- ACC2: 79.12% (-3.35% vs baseline)
- Best val ACC2: 84.26%, test collapse → overfitting confirmed
- Verdict: ❌ Closed

## Root Cause
MOSI has only 1284 training samples. High-dimensional frozen features (1024d) provide more capacity than the dataset can support, leading to severe overfitting. The base models (768d) are actually optimal for this data scale.

## Next Direction
Text backbone fine-tuning (RoBERTa-large / DeBERTa-v3) on raw transcripts, which can learn dataset-specific representations.
