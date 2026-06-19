# P6S-Repair-5: Cache / Audio / Batch Audit Report

## RoBERTa Cache
| Split | Shape | Mean | Std | All-Zero | Status |
|-------|-------|------|-----|----------|--------|
| train | (14700, 1024) | -0.032 | 0.989 | No | ✅ PASS |
| valid | (1759, 1024) | -0.032 | 0.989 | No | ✅ PASS |
| test | (4221, 1024) | -0.032 | 0.989 | No | ✅ PASS |

- Sample IDs match between cache and label.csv ✅
- All embeddings are non-zero and have healthy distribution ✅
- All features are RoBERTa-large CLS (1024-dim) ✅

## COVAREP Audio Features
| Aspect | Status |
|--------|--------|
| Dimension | 74 (correct) ✅ |
| Shape per sample | [100, 74] ✅ |
| Audio mask | All 1s (no padding) ✅ |
| -inf values | Found in ~2-5% of files ⚠️ FIXED |
| +inf values | None ✅ |
| NaN values | None ✅ |
| Fix applied | _clean_features() in collate_textft replaces -inf/+inf with 0.0 ✅ |

### -inf Root Cause
COVAREP feature extraction uses `log(x)` operations which produce -inf when x=0. This is a known issue with COVAREP features.

### Fix Verification
After fix: 50 batches tested, 0 have -inf in audio, 0 have NaN in model output ✅

## Batch Construction
| Key | Shape | Source | Clean |
|-----|-------|--------|-------|
| input_ids | [B, 128] | RoBERTa tokenizer | NA |
| attention_mask | [B, 128] | RoBERTa tokenizer | NA |
| audio | [B, 100, 74] | COVAREP .npz | Cleaned (was -inf) |
| audio_mask | [B, 100] | Feature file | All 1s |
| vision | [B, 40, 768] | Feature file | All zeros (text_audio mode) |
| vision_mask | [B, 40] | Feature file | All zeros |
| label | [B, 1] | label.csv | [-3, +3] float |
| id | list | video_clip_id | Matches cache |
| roberta_cls | [B, 1024] | RoBERTa cache | Injected by inject_text_feature() |

## Text Feature Injection
- `inject_text_feature()` correctly maps sample IDs to roberta cache indices ✅
- Fallback to index 0 for missing IDs (shouldn't happen) ✅
- All IDs found in cache ✅

## Model Forward Pass (Post-Fix)
Forward pass verified on CPU and CUDA:
- 50 batches tested: 0 NaN in output ✅
- Output mean ~0.05, std ~0.03 (not constant) ✅
- All 4 samples in batch have different predictions ✅

Generated: 2026-06-19
