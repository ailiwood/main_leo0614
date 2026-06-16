# Project Status — 基于扩展 LSTM 与 Transformer 的跨模态情感分析研究

**Last update**: 2026-06-16 P4S

## Current Phase
P4S — Standard sequence feature access + strict evaluation protocol repair

## What's Trustworthy
- P4A: Internal dev reference only (test leakage detected)
- P4R: Audit complete — root cause identified (T=1 features)
- All P0-P3 code/modules: Verified ✅

## What's NOT Paper-Ready
- P4A 75.25% result (test leakage, T=1 limitation)
- Any P3B candidate module results (T=1)
- Any P3C diagnostic results (T=1)

## Current Main Model
C0: sLSTM ×3 + AWAF + regression/classification heads (3.06M params)
No DEConv/CME/Data2Vec in main model.

## Data/Feature Status
- MOSI TMDC-v1: T=1 pooled (dev use only)
- MOSI sequence: downloading from MLCL...
- MOSEI: vision features missing

## How to Run Strict Protocol Smoke Test
```
python -c "from engine.strict_trainer import StrictTrainer; ..."
```

## Environment
mme_xlstm_stable: Python 3.10, PyTorch 2.11.0+cu128 (STABLE), RTX 5070 Ti

## Next Steps
1. Complete sequence feature download
2. Sequence feature smoke test
3. MOSI strict protocol C0 retrain
4. Compare T=1 vs sequence results
