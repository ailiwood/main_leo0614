# P6T-PreFreeze: MOSEI Text-Audio-Vision Candidate Report

**Date**: 2026-06-19  
**Status**: ⚠️ Config ready, smoke pending (GPU occupied by text_audio training)

## 1. TAV Configuration
- Config: `configs/experiments/p6t_prefreeze_mosei_mainline/mosei_main_text_audio_vision_cached_s42.yaml`
- Mode: `text_av_residual`
- Audio dim: 74 (COVAREP)
- Vision dim: 768 (FACET 40-frame)
- Hidden dim: 256
- AWAF: enabled (3-modality weights: w_t, w_a, w_v)
- LoRA: r=16, targets=['query', 'value']

## 2. Vision Feature Status (Pre-run Assessment)

| Check | Status |
|-------|--------|
| Vision feature files exist | ✅ (data/processed/mosei_full/train/*.npz) |
| Vision shape | [40, 768] per sample |
| Vision content | ⚠️ **All zeros** in sampled files |
| Vision mask | All zeros |
| Likely cause | MOSEI vision features not extracted (FACET unavailable on this system) |

**Critical finding**: MOSEI vision features are all zeros. This means:
- TAV mode will have zero vision input → AWAF will learn w_v ≈ 0
- TAV will effectively be identical to text_audio mode
- No multimodal gain from vision

## 3. Decision

**TAV full training is NOT recommended for MOSEI** because:
1. Vision features are all zeros — no information content
2. Training TAV will waste GPU time learning to ignore vision
3. TAV will converge to same result as text_audio

### Recommended Approach for Thesis
- **Primary result**: text_audio (has real audio features)
- **Diagnostic result**: text_only (establishes text-only baseline)
- **Vision discussion**: Note that FACET vision features are unavailable for MOSEI; vision modality left as candidate for future work or for MOSI dataset (which has real vision features)

## 4. Smoke Test Command (ready when GPU free)
```bash
python scripts/train_textft_lora_mainline.py \
  --config configs/experiments/p6t_prefreeze_mosei_mainline/mosei_main_text_audio_vision_cached_s42.yaml \
  --device cuda --smoke
```

Expected outcome: Model will run without NaN/crash, but vision weights will be near-zero.

## 5. Alternative: MOSI TAV
MOSI has real vision features (768d, non-zero). P6K `text_av_residual` achieved 86.13% ACC2 on MOSI. If TAV proof needed, use MOSI results.

Generated: 2026-06-19
