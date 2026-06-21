# P6AD-G1: MOSI TAV Go/No-Go Decision

**Date**: 2026-06-21  
**Decision**: **GO** ✅ — Proceed to G2 validation

---

## Audit Results

| Criterion | Result |
|-----------|--------|
| Vision features exist | ✅ CLIP-L14, 32×1024 |
| Vision is sequence (not pooled) | ✅ 32 time steps |
| Vision has valid frames | ✅ All 32 per sample |
| Vision NaN/Inf/zero | ✅ None |
| Audio features exist | ✅ data2vec, 100×768 |
| Audio has valid frames | ✅ |
| Sample IDs align with labels | ✅ 2,199/2,199 |
| Split assignment correct | ✅ train/val/test |
| No split leakage | ✅ |
| Compatible with TextFTMultimodalDataset | ✅ |
| Compatible with TAV model interface | ✅ (with dim overrides) |

## Feature Contract

```yaml
mosi_tav_v1:
  vision_source: CLIP-L14
  vision_dim: 1024
  vision_seq_len: 32
  audio_source: data2vec
  audio_dim: 768
  audio_seq_len: 100
  text_source: raw → RoBERTa-large
  total_samples: 2199
  train: 1284
  val: 229
  test: 686
```

## Required Config Overrides (vs MOSEI)

```yaml
model:
  audio_dim: 768     # MOSEI: 74
  vision_dim: 1024   # MOSEI: 713
data:
  csv_path: data/mosi/label.csv
  dataset: mosi
  feature_root: data/processed/mosi_tav_v1
  data_version: mosi_tav_v1
  cohort_id: mosi_official_tav_complete_case_v1
```

## Risks

| Risk | Mitigation |
|------|------------|
| MOSI small sample (2,199) may cause instability | Overfit200 validation before formal training |
| Different feature sources from MOSEI | Clearly document; do not claim same features |
| CLIP-L14 1024-dim may overfit with few samples | Use LoRA + freeze text; low LR |
| data2vec 768-dim audio may dominate small vision | AWAF weights should balance |

## Next Steps

1. P6AD-G2: Static validation (model instantiation)
2. P6AD-G2: Dynamic validation (gradient flow)
3. P6AD-G2: Overfit200 test (training stability)
4. P6AD-G3: MOSI TAV main model (if G2 passes)
