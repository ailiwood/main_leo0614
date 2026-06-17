# HANDOFF_PHASE_08Y.md — P4Y Data Audit & Rebuild

## Key Findings

### MOSI: Grade B+ — KEEP
- 2199 MP4 + 2199 WAV + labels ✅ Complete
- Rare asset: most researchers don't have MOSI MP4 files
- Enables Vision T>1 extraction (CLIP-ViT per-frame)
- Decision: **KEEP, no redownload needed**

### MOSEI: Grade C → B (after fix)
- 8636 WAV + CSV labels ✅
- **0 MP4 files — NOT available from CMU** (YouTube privacy restriction)
- Old .features files: **DELETED** (0-1 labels, inconsistent dimensions)
- Standard approach: use CMU-MultimodalSDK .csd computational sequences
- Decision: **DOWNLOAD SDK features** (COVAREP+FACET+GloVe)
- MOSEI vision via FACET (35D), NOT CLIP-ViT

### Critical Discovery
**MOSEI raw videos are NOT publicly available.** CMU cannot share them due to YouTube creator privacy. The official dataset provides pre-extracted computational sequences (.csd files). Most papers use these features (COVAREP audio + FACET vision + GloVe text), not raw frames.

## Actions Taken
1. Deleted old MOSEI .features (892+108+263 MB) — unreliable format
2. Keeping MOSI MP4 data — valuable for Vision T>1
3. Keeping MOSEI Audio + Labels — useful for SDK alignment
4. Documented official sources and download plan

## Download Plan
- MOSEI: CMU-MultimodalSDK → .csd files (COVAREP + FACET + GloVe + sentiment labels)
- GitHub: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK

## GitHub
- Branch: p4y-data-redownload-audit
- URL: https://github.com/ailiwood/main_leo0614
