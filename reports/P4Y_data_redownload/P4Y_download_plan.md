# P4Y Download Plan

## MOSI: KEEP current data (no redownload needed)
- 2199 MP4 + 2199 WAV + labels ✅
- Current data is complete and rare (most researchers don't have MP4)
- Delete: nothing. Keep: all.

## MOSEI: Download CMU-MultimodalSDK computational sequences

### Source
- GitHub: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
- License: Research use (CMU dataset terms)
- Provides: .csd (HDF5) files with COVAREP + FACET + GloVe + labels

### What to download
```
cd data/raw/mosei_sdk/
python -c "
from mmsdk import mmdatasdk
cmumosei = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel, 'data/raw/mosei_sdk/')
"
```

### What MOSEI SDK provides
- COVAREP: acoustic features (74-dim)
- FACET 4.2: visual features (35-dim facial action units)  
- GloVe: text word embeddings (300-dim)
- Labels: sentiment [-3,+3]

### Note on Vision
- FACET features are computational sequences, NOT raw videos
- T>1 is possible (frame-level features)
- But these are low-dimensional (35D FACET vs 768D CLIP)
- This is what MMSA/MLCL papers use as "standard features"

## Delete old MOSEI .features files
- train.features (892MB), val.features (108MB), test.features (263MB)
- Reason: 0-1 binary labels, inconsistent dimensions across splits
- These are NOT standard MOSEI features
