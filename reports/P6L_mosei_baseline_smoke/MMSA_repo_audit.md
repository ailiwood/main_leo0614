# P6L MMSA Classic Baseline Audit

- **Repo**: https://github.com/thuiar/MMSA
- **Commit**: `a94e65d`
- **License**: MIT
- **Local**: `external/MMSA/`
- **Supported models**: TFN, LMF, MulT, MISA, Self-MM, MMIM (+ more)
- **Has MOSI**: ✅ (built-in data loading)
- **Has MOSEI**: ✅ (built-in data loading)
- **Data format**: .pkl files per modality (text_glove, audio, vision)
- **Train script**: `src/train.py` (model-specific configs)
- **Metrics**: custom metrics module, needs verification against our `utils/metrics.py`
- **Per-sample prediction**: ✅ (model outputs available)

## Smoke Status: B

Code is well-structured but needs:
1. Environment setup (Python 3.8, torch 1.x)
2. MOSI data in MMSA format (.pkl files)
3. Config adaptation for our feature version
4. Metric cross-validation with `utils/metrics.py`

## Classic Baseline Matrix

| Model | In MMSA | MOSI | MOSEI | Priority |
|-------|:-------:|:----:|:-----:|:--------:|
| TFN | ✅ | ✅ | ✅ | P1 |
| LMF | ✅ | ✅ | ✅ | P1 |
| MulT | ✅ | ✅ | ✅ | P0 |
| MISA | ✅ | ✅ | ✅ | P0 |
| Self-MM | ✅ | ✅ | ✅ | P0 |
| MMIM | ✅ | ✅ | ✅ | P0 |

## Next Steps

1. Set up `mme_mmsa` conda env
2. Prepare MOSI features in MMSA format
3. Run Self-MM smoke (3 epochs) — highest priority classic baseline
4. Cross-validate metrics with `utils/metrics.py`
