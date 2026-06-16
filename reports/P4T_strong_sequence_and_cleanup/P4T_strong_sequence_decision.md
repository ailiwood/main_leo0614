# P4T Strong Sequence Decision

## Cleanup Status
✅ File classification done. No files deleted. Awaiting user confirmation for archiving.

## GitHub Prep
✅ .gitignore updated. README_PROJECT_STATUS.md written.

## Strong Sequence Features
⏳ Extraction script written. Models downloading (DeBERTa 1.63GB + wav2vec2 1.26GB).
Target: token-level DeBERTa 1024d + frame-level wav2vec2 1024d + CLIP ViT per-frame.
Expected T: text~50, audio~100, vision~20.

## C0 Strict Retraining
Blocked on feature extraction completion.

## Recommendations
1. **Wait for model download + extraction** (est. 2-4 hours for full MOSI)
2. **OR** accept TMDC-v1 T=1 as the primary feature set with honesty about T=1 limitation
3. **OR** use BERT-base instead of DeBERTa-large for faster extraction
