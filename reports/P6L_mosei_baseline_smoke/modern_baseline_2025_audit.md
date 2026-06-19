# P6L 2025 Modern Baseline Audit

## MLCL (Priority P0)

- **Repo**: https://github.com/YetZzzzzz/MLCL
- **Commit**: `7cc0f25`
- **License**: MIT
- **Local**: `external/MLCL/`
- **Has MOSI**: ✅ (mosi_mcl.pkl at `external/features_mlcl/`)
- **Has MOSEI**: ❌ (no mosei_mcl.pkl found)
- **Train script**: `MCLC_MSA/main_MLCL.py`
- **Config**: argparse-based
- **Data format**: custom .pkl files (list of dicts with text/audio/vision/label)
- **Metrics**: sklearn-based (ACC2, F1, MAE, Corr)
- **Per-sample prediction**: ✅ (regression output available)
- **Smoke status**: B — can import and load features, needs env setup
- **Risk**: No MOSEI features; MOSI features use different preprocessing than ours
- **Next**: Set up env, run 3ep smoke on MOSI

## DLF (Priority P1)

- **Repo**: https://github.com/pwang322/DLF
- **Commit**: `251bf2c`
- **License**: MIT
- **Local**: `external/DLF/`
- **Has MOSI**: ✅ (configurable)
- **Has MOSEI**: ✅ (configurable)
- **Train script**: `run.py` → `train.py`
- **Config**: `config.py` (class-based)
- **Data format**: MMSA format (text/audio/vision .pkl files)
- **Smoke status**: B — code structure clear, depends on MMSA data
- **Risk**: Requires MMSA data format; may need path adaptation
- **Next**: Check MMSA data interface compatibility

## DPDF-LQ (Priority P1)

- **Repo**: https://github.com/ZhouMiaoGX/DPDF-LQ
- **Commit**: `5cebfb9`
- **License**: MIT
- **Local**: `external/DPDF-LQ/`
- **Has MOSI**: ✅
- **Has MOSEI**: ✅
- **Train script**: `train.py`
- **Config**: YAML-based
- **Data format**: custom preprocessed features
- **Smoke status**: B — code available, needs data download
- **Risk**: May require downloading processed data from external source
- **Next**: Check README for data download instructions

## Not Tested This Round

| Model | Reason |
|-------|--------|
| CASP | TTA method → separate table, not mixed with end-to-end baselines |
| DashFusion | Out of scope for P6L (baseline convergence) |
| R3DG | Deferred to later phase |
