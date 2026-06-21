# P6AD-G1: MOSI TAV Feature Contract Audit

**Date**: 2026-06-21  
**MOSI_TAV_STATUS**: `ready` ✅

---

## 1. Source Data

| Property | Value |
|----------|-------|
| Source directory | `data/features_strong_sequence_mosi_v6_vision_l14_full/` |
| Label CSV | `data/mosi/label.csv` (2,199 rows) |
| Vision source | CLIP-L14 (DINOv2/CLIP large) |
| Audio source | data2vec |
| Text source | Raw text in label CSV → RoBERTa-large tokenizer |

## 2. Feature Dimensions

| Modality | Shape | Dim | Frames | Type |
|----------|-------|-----|--------|------|
| Vision | (32, 1024) | 1024 | 32 | CLIP-L14 frame embeddings |
| Audio | (100, 768) | 768 | 100 | data2vec |
| Text | raw | — | — | Tokenized by RoBERTa-large |

## 3. Split Distribution

| Split | Samples | Vision All-Zero | Audio All-Zero | NaN/Inf |
|-------|---------|-----------------|----------------|---------|
| train | 1,284 | 0 | 0 | 0 |
| val | 229 | 0 | 0 | 0 |
| test | 686 | 0 | 0 | 0 |
| **Total** | **2,199** | **0** | **0** | **0** |

## 4. Alignment Check

| Check | Result |
|-------|--------|
| Sample IDs match label CSV | ✅ 2,199/2,199 (100%) |
| Missing feature files | 0 |
| Extra feature files | 0 |
| Split assignment matches label CSV | ✅ |
| No train/val/test overlap | ✅ |
| All vision_mask have valid frames | ✅ (all 32 frames valid per sample) |
| All audio_mask have valid frames | ✅ |

## 5. Model Interface Compatibility

| Check | Status |
|-------|--------|
| Vision is sequence (not pooled) | ✅ 32 time steps |
| Vision mask available | ✅ |
| Audio mask available | ✅ |
| Standard .npz format compatible with TextFTMultimodalDataset | ✅ |
| Sample ID format consistent | ✅ `{video_id}_{clip_id}` |

## 6. Feature Version Differences from MOSEI

| Property | MOSEI TAV | MOSI TAV | Compatible? |
|----------|-----------|----------|-------------|
| Vision source | OpenFace2 | CLIP-L14 | ❌ Different |
| Vision dim | 713 | 1024 | ❌ Different |
| Vision seq len | 50 | 32 | ❌ Different |
| Audio source | COVAREP | data2vec | ❌ Different |
| Audio dim | 74 | 768 | ❌ Different |
| Text | Raw → RoBERTa | Raw → RoBERTa | ✅ Same |

**These MUST be documented as different feature versions.** MOSI uses `mosi_tav_v1` (CLIP-L14 + data2vec), while MOSEI uses `mosei_tav_openface2_v1` (OpenFace2 + COVAREP).

## 7. Go/No-Go Decision

**GO**: MOSI TAV data is ready for training.
- All 2,199 samples have valid text, audio, and vision
- No NaN, Inf, or all-zero features
- Perfect alignment with label CSV
- Compatible with TextFTMultimodalDataset interface
- Vision is a proper sequence (32 time steps) supporting sLSTM temporal modeling

**Caveats**:
- Vision features are CLIP-L14 (1024-dim), NOT OpenFace2 (713-dim) — different feature version
- Audio features are data2vec (768-dim), NOT COVAREP (74-dim) — different feature version
- Model config MUST use `vision_dim: 1024` and `audio_dim: 768` for MOSI
- Results are NOT directly comparable to MOSEI TAV results at the feature level
